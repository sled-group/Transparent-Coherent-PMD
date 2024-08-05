from datasets import Dataset
from huggingface_hub import whoami
import math
import numpy as np
from pprint import pprint
import torch
import time
from typing import List, Optional, Union, Callable
from transformers import (
    PreTrainedTokenizerBase,
)
from trl import PPOConfig
from trl.core import (
    WANDB_PADDING,
    PPODecorators,
    convert_to_scalar,
    logprobs_from_logits,
    stack_dicts,
    stats_to_np,
)
from trl.models import unwrap_model_for_generation, PreTrainedModelWrapper
from trl.trainer import PPOTrainer
import typing

class PerTokenPPOTrainer(PPOTrainer):
   
    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        model: Optional[PreTrainedModelWrapper] = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        training_data_collator: Optional[typing.Callable] = None,
    ):
        """
        Initialize PPOTrainer.

        Args:
            config (`PPOConfig`):
                Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for KL penalty
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face tokenizer
            dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model. If none is passed,
                a warning will be raised in a multi-GPU setting.
            optimizer (Optional[`torch.optim.Optimizer`]):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function that is going to be used for `prepare_dataloader` method. Note this collator
                is different from the one we use for training. Pass a valid `training_data_collator` instead.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared.
                used only if `ref_model` is `None`.
            lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
                Learning rate scheduler used for training.
            training_data_collator (Optional[function]):
                Custom data collator used for training.
        """
        super().__init__(config, model, ref_model, tokenizer, dataset, optimizer, data_collator, num_shared_layers, lr_scheduler, training_data_collator)
        # Superclass sets ref_model to None when a peft model is being trained, but even with a peft model we want to be able to use a separate reference model
        if ref_model is not None and self.ref_model is None:
            self.ref_model = ref_model

    """PPOTrainer adapted from and overriding Hugging Face's version (https://github.com/huggingface/trl/blob/98ad01ddfd1e1b67ec018014b83cba40e0caea66/trl/trainer/ppo_trainer.py) to be able to assign scores from reward model for specific token indices."""
    def _generate_batched(
        self,
        model: PreTrainedModelWrapper,
        query_tensors: List[torch.Tensor],
        length_sampler: Optional[Callable] = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        outputs = []

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            # hack: don't use PEFT functionality in PPO trainer, since there seems to be an issue with LoRA models
            with unwrap_model_for_generation(model, self.accelerator, is_peft_model=False) as unwrapped_model:
                generations = unwrapped_model.generate(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not return_prompt and not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt

                if remove_padding and self.tokenizer.eos_token_id in output:
                    pad_mask = output == self.tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                    output = output[: pad_start + 1]  # keep the eos token at the end

                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Optional[Callable] = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        generate_ref_response: bool = False,
        **generation_kwargs,
    ):
        """
        Generate response with the model given the query tensor.
        call the `generate` method of the model.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`seq_len`) containing query tokens or a list of tensors of shape (`seq_len`).
            length_sampler (`Callable`, *optional*):
                Callable that returns the number of newly generated tokens.
            batch_size (`int`, *optional):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.
            generate_ref_response (`bool`, *optional*):
                If set to `True` the reference response is also generated, defaults to `False`.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """
        if generate_ref_response:
            if self.ref_model is None:
                raise NotImplementedError("PerTokenPPOTrainer requires a ref_model to be passed, as when the base model is a PEFT model, the PPOTrainer class inadvertently updates the underlying model during training.")
            ref_model = self.ref_model

        if isinstance(query_tensor, List):
            query_tensor = [qt.to(self.current_device) for qt in query_tensor]
            response = self._generate_batched(
                self.model,
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )
            if generate_ref_response:
                query_tensor = [qt.to(ref_model.device) for qt in query_tensor]
                ref_response = self._generate_batched(
                    ref_model,
                    query_tensor,
                    length_sampler=length_sampler,
                    batch_size=batch_size,
                    return_prompt=return_prompt,
                    **generation_kwargs,
                )

        else:
            if len(query_tensor.shape) == 2:
                raise ValueError(
                    "query_tensor must be a tensor of shape (`seq_len`) or a list of tensors of shape (`seq_len`)"
                )

            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            query_tensor = query_tensor.to(self.current_device)
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                response = unwrapped_model.generate(input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs)

            if generate_ref_response:
                query_tensor = query_tensor.to(ref_model.device)
                # Hack: don't disable any PEFT adapter here since it doesn't seem to work
                with unwrap_model_for_generation(
                    ref_model, self.accelerator, is_peft_model=False
                ) as unwrapped_model:
                    ref_response = unwrapped_model.generate(
                        input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
                    )

            if not return_prompt and not self.is_encoder_decoder:
                response = response[:, query_tensor.shape[0] :]
                if generate_ref_response:
                    ref_response = ref_response[:, query_tensor.shape[0] :]

        if generate_ref_response:
            return response, ref_response
        return response

    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        model.eval()

        return_values = True
        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            results = model(**input_kwargs)
            if len(results) == 3:
                logits, _, values = results
            else:
                logits = results[0]
                values = None
                return_values = False

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1  # logprobs starts from the second query token
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1] if return_values else None,
            torch.cat(all_masks)[:, :-1],
        )

    def _step_safety_checker(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
        score_indices: List[torch.FloatTensor],
        masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            masks (List[`torch.LongTensor`], *optional*):
                list of optional tensors containing the masks of shape (`query_length` + `response_length`)
        Returns:
            `tuple`: The input processed data.
        """
        assert len(scores) == batch_size

        for name, tensor_list in zip(["queries", "responses"], [queries, responses]):
            if not isinstance(tensor_list, list):
                raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(f"Elements in {name} must be tensors - got {type(tensor_list[0])}")
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
                )

        for score, score_idxs in zip(scores, score_indices):
            assert score.shape[0] >= int(torch.max(score_idxs).cpu().numpy()) + 1

        # add queries, scores and responses on the correct device
        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = scores.to(self.current_device)
        masks = [tensor.to(self.current_device) for tensor in masks] if masks is not None else None

        # squeeze scores if needed
        for i, score in enumerate(scores):
            if score.dim() > 1:
                raise ValueError(f"Scores must be 1-dimensional - got {score.dim()} for {score}")
            elif score.dim() == 1:
                scores[i] = score.squeeze()

        return queries, responses, scores, masks

    @PPODecorators.empty_device_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: torch.FloatTensor,
        score_indices: List[torch.LongTensor],
        response_masks: Optional[List[torch.LongTensor]] = None,
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                Tensors containing the scores to apply to the sequence, shape (`batch_size`, `n`).
            score_indices (`torch.FloatTensor`)):
                Tensor of shape (`batch_size`, `n`) indicating at which response token indices to apply `scores` in rewards.
            response_masks (List[`torch.FloatTensor`], *optional*)):
                List of tensors containing masks of the response tokens.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        if self.ref_model is None:
            raise NotImplementedError("PerTokenPPOTrainer requires a ref_model to be passed, as when the base model is a PEFT model, the PPOTrainer class inadvertently updates the underlying model during training.")

        bs = self.config.batch_size

        queries, responses, scores, response_masks = self._step_safety_checker(
            bs, queries, responses, scores, score_indices, response_masks
        )
        if self.config.use_score_scaling:
            raise NotImplementedError("Score scaling not implemented.")
        #     # Score scaling
        #     scores_mean, scores_std = self.running.update(scores)
        #     tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
        #     score_scaling_factor = self.running.std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
        #     if self.config.use_score_norm:
        #         scores = (scores - self.running.mean.to(**tensor_to_kwargs)) / score_scaling_factor
        #     else:
        #         scores /= score_scaling_factor

        if self.config.score_clip is not None:
            raise NotImplementedError("Score clipping not implemented.")
        #     # Score clipping
        #     scores_dtype = scores.dtype
        #     scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = torch.mean(scores)
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)
        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_input_ids"],
                    dim=1,
                    pad_index=self.tokenizer.pad_token_id,
                    pad_first=pad_first,
                )
                model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                    model_inputs["decoder_attention_mask"],
                    dim=1,
                    pad_index=0,
                    pad_first=pad_first,
                )
        model_inputs_names = list(model_inputs.keys())

        # Pad score indices to make sure rewards are aligned correctly
        score_indices = [(torch.tensor([self.tokenizer.pad_token_id] * (model_inputs['input_ids'].shape[-1] - si.shape[-1])).to(si.device).long(), si) for si in score_indices]
        score_indices = [torch.cat((pads, si), dim=0) if self.tokenizer.padding_side == "left" else torch.cat((si, pads), dim=0) for pads, si in score_indices]

        full_kl_penalty = self.config.kl_penalty == "full"

        with torch.inference_mode():
            all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                responses,
                model_inputs,
                response_masks=response_masks,
                return_logits=full_kl_penalty,
            )
            with self.optional_peft_ctx():
                ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                    self.ref_model,
                    queries,
                    responses,
                    model_inputs,
                    return_logits=full_kl_penalty,
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        with torch.inference_mode():
            t = time.time()
            if full_kl_penalty:
                active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
                ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

                rewards, non_score_reward, kls = self.compute_rewards(
                    scores, score_indices, active_full_logprobs, ref_full_logprobs, masks
                )
            else:
                rewards, non_score_reward, kls = self.compute_rewards(scores, score_indices, all_logprobs, ref_logprobs, masks)
            timing["time/ppo/compute_rewards"] = time.time() - t

            t = time.time()
            values, advantages, returns = self.compute_advantages(values, rewards, masks)
            timing["time/ppo/compute_advantages"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
        }
        batch_dict.update(model_inputs)

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            b_inds = np.random.permutation(bs)
            for backward_batch_start in range(0, bs, self.config.backward_batch_size):
                backward_batch_end = backward_batch_start + self.config.backward_batch_size
                backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

                for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                    mini_batch_end = mini_batch_start + self.config.mini_batch_size
                    mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                    mini_batch_dict = {
                        "logprobs": batch_dict["logprobs"][mini_batch_inds],
                        "values": batch_dict["values"][mini_batch_inds],
                        "masks": batch_dict["masks"][mini_batch_inds],
                        # hacks: the queries and responses are ragged.
                        "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                        "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                        "advantages": batch_dict["advantages"][mini_batch_inds],
                        "returns": batch_dict["returns"][mini_batch_inds],
                    }
                    for k in model_inputs_names:
                        mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                    with self.accelerator.accumulate(self.model):
                        model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                        logprobs, logits, vpreds, _ = self.batched_forward_pass(
                            self.model,
                            mini_batch_dict["queries"],
                            mini_batch_dict["responses"],
                            model_inputs,
                            return_logits=True,
                        )
                        train_stats = self.train_minibatch(
                            mini_batch_dict["logprobs"],
                            mini_batch_dict["values"],
                            logprobs,
                            logits,
                            vpreds,
                            mini_batch_dict["masks"],
                            mini_batch_dict["advantages"],
                            mini_batch_dict["returns"],
                        )
                        all_stats.append(train_stats)

            # typically, early stopping is done at the epoch level
            if self.config.early_stopping:
                policykl = train_stats["policy/policykl"]
                early_stop = self._early_stop(policykl)
                if early_stop:
                    break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
        train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
            kls=kls,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return stats
    
    # @PPODecorators.empty_device_cache()
    # def step_query_only(
    #     self,
    #     queries: List[torch.LongTensor],
    #     scores: torch.FloatTensor,
    #     score_indices: List[torch.LongTensor],
    #     response_masks: Optional[List[torch.LongTensor]] = None,
    # ):
    #     """
    #     Run a PPO optimisation step given a list of queries, model responses, and rewards.

    #     Args:
    #         queries (List[`torch.LongTensor`]):
    #             List of tensors containing the encoded queries of shape (`query_length`)
    #         responses (List[`torch.LongTensor`]):
    #             List of tensors containing the encoded responses of shape (`response_length`)
    #         scores (List[`torch.FloatTensor`]):
    #             Tensors containing the scores to apply to the sequence, shape (`batch_size`, `n`).
    #         score_indices (`torch.FloatTensor`)):
    #             Tensor of shape (`batch_size`, `n`) indicating at which response token indices to apply `scores` in rewards.
    #         response_masks (List[`torch.FloatTensor`], *optional*)):
    #             List of tensors containing masks of the response tokens.

    #     Returns:
    #         `dict[str, Any]`: A summary of the training statistics
    #     """
    #     if self.ref_model is None:
    #         raise NotImplementedError("PerTokenPPOTrainer requires a ref_model to be passed, as when the base model is a PEFT model, the PPOTrainer class inadvertently updates the underlying model during training.")

    #     bs = self.config.batch_size

    #     queries = [tensor.to(self.current_device) for tensor in queries]
    #     scores = scores.to(self.current_device)
    #     masks = [tensor.to(self.current_device) for tensor in masks] if masks is not None else None

    #     # squeeze scores if needed
    #     for i, score in enumerate(scores):
    #         if score.dim() > 1:
    #             raise ValueError(f"Scores must be 1-dimensional - got {score.dim()} for {score}")
    #         elif score.dim() == 1:
    #             scores[i] = score.squeeze()

    #     timing = dict()
    #     t0 = time.time()

    #     t = time.time()

    #     input_ids = [q for q in queries]
    #     model_inputs = self.data_collator(
    #         [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
    #     ).to(self.current_device)
    #     model_inputs.pop("labels", None)  # we don't want to compute LM losses

    #     if self.is_distributed:
    #         pad_first = self.tokenizer.padding_side == "left"

    #         model_inputs["input_ids"] = self.accelerator.pad_across_processes(
    #             model_inputs["input_ids"],
    #             dim=1,
    #             pad_index=self.tokenizer.pad_token_id,
    #             pad_first=pad_first,
    #         )
    #         model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
    #             model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
    #         )
    #         if self.is_encoder_decoder:
    #             model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
    #                 model_inputs["decoder_input_ids"],
    #                 dim=1,
    #                 pad_index=self.tokenizer.pad_token_id,
    #                 pad_first=pad_first,
    #             )
    #             model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
    #                 model_inputs["decoder_attention_mask"],
    #                 dim=1,
    #                 pad_index=0,
    #                 pad_first=pad_first,
    #             )
    #     model_inputs_names = list(model_inputs.keys())

    #     # Pad score indices to make sure rewards are aligned correctly
    #     score_indices = [(torch.tensor([self.tokenizer.pad_token_id] * (model_inputs['input_ids'].shape[-1] - si.shape[-1])).to(si.device).long(), si) for si in score_indices]
    #     score_indices = [torch.cat((pads, si), dim=0) if self.tokenizer.padding_side == "left" else torch.cat((si, pads), dim=0) for pads, si in score_indices]

    #     full_kl_penalty = self.config.kl_penalty == "full"

    #     with torch.inference_mode():
    #         all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
    #             self.model,
    #             queries,
    #             responses,
    #             model_inputs,
    #             response_masks=response_masks,
    #             return_logits=full_kl_penalty,
    #         )
    #         with self.optional_peft_ctx():
    #             ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
    #                 self.ref_model,
    #                 queries,
    #                 responses,
    #                 model_inputs,
    #                 return_logits=full_kl_penalty,
    #             )

    #     timing["time/ppo/forward_pass"] = time.time() - t

    #     with torch.inference_mode():
    #         t = time.time()
    #         if full_kl_penalty:
    #             active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
    #             ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

    #             rewards, non_score_reward, kls = self.compute_rewards(
    #                 scores, score_indices, active_full_logprobs, ref_full_logprobs, masks
    #             )
    #         else:
    #             rewards, non_score_reward, kls = self.compute_rewards(scores, score_indices, all_logprobs, ref_logprobs, masks)
    #         timing["time/ppo/compute_rewards"] = time.time() - t

    #         t = time.time()
    #         values, advantages, returns = self.compute_advantages(values, rewards, masks)
    #         timing["time/ppo/compute_advantages"] = time.time() - t

    #     # upcast to float32 to avoid dataset issues
    #     batch_dict = {
    #         "queries": queries,
    #         "responses": responses,
    #         "logprobs": all_logprobs.to(torch.float32),
    #         "values": values.to(torch.float32),
    #         "masks": masks,
    #         "advantages": advantages,
    #         "returns": returns,
    #     }
    #     batch_dict.update(model_inputs)

    #     t = time.time()
    #     all_stats = []
    #     early_stop = False
    #     for _ in range(self.config.ppo_epochs):
    #         if early_stop:
    #             break
    #         b_inds = np.random.permutation(bs)
    #         for backward_batch_start in range(0, bs, self.config.backward_batch_size):
    #             backward_batch_end = backward_batch_start + self.config.backward_batch_size
    #             backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

    #             for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
    #                 mini_batch_end = mini_batch_start + self.config.mini_batch_size
    #                 mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
    #                 mini_batch_dict = {
    #                     "logprobs": batch_dict["logprobs"][mini_batch_inds],
    #                     "values": batch_dict["values"][mini_batch_inds],
    #                     "masks": batch_dict["masks"][mini_batch_inds],
    #                     # hacks: the queries and responses are ragged.
    #                     "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
    #                     "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
    #                     "advantages": batch_dict["advantages"][mini_batch_inds],
    #                     "returns": batch_dict["returns"][mini_batch_inds],
    #                 }
    #                 for k in model_inputs_names:
    #                     mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
    #                 with self.accelerator.accumulate(self.model):
    #                     model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

    #                     logprobs, logits, vpreds, _ = self.batched_forward_pass(
    #                         self.model,
    #                         mini_batch_dict["queries"],
    #                         mini_batch_dict["responses"],
    #                         model_inputs,
    #                         return_logits=True,
    #                     )
    #                     train_stats = self.train_minibatch(
    #                         mini_batch_dict["logprobs"],
    #                         mini_batch_dict["values"],
    #                         logprobs,
    #                         logits,
    #                         vpreds,
    #                         mini_batch_dict["masks"],
    #                         mini_batch_dict["advantages"],
    #                         mini_batch_dict["returns"],
    #                     )
    #                     all_stats.append(train_stats)

    #         # typically, early stopping is done at the epoch level
    #         if self.config.early_stopping:
    #             policykl = train_stats["policy/policykl"]
    #             early_stop = self._early_stop(policykl)
    #             if early_stop:
    #                 break

    #     timing["time/ppo/optimize_step"] = time.time() - t

    #     t = time.time()
    #     train_stats = stack_dicts(all_stats)

    #     # reshape advantages/ratios such that they are not averaged.
    #     train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
    #     train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
    #     train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

    #     stats = self.record_step_stats(
    #         scores=scores,
    #         logprobs=all_logprobs,
    #         ref_logprobs=ref_logprobs,
    #         non_score_reward=non_score_reward,
    #         train_stats=train_stats,
    #         kl_coef=self.kl_ctl.value,
    #         masks=masks,
    #         queries=queries,
    #         responses=responses,
    #         kls=kls,
    #     )
    #     # Gather/Reduce stats from all processes
    #     if self.is_distributed:
    #         stats = self.gather_stats(stats)
    #     stats = stats_to_np(stats)
    #     timing["time/ppo/calc_stats"] = time.time() - t
    #     stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

    #     # Update the KL control - multiply the batch_size by the number of processes
    #     self.kl_ctl.update(
    #         stats["objective/kl"],
    #         self.config.batch_size * self.accelerator.num_processes,
    #     )

    #     # Log the total ppo time
    #     timing["time/ppo/total"] = time.time() - t0
    #     stats.update(timing)

    #     # post-process stats for tensorboard and other loggers
    #     if self.config.log_with != "wandb":
    #         stats = convert_to_scalar(stats)

    #     if self.lr_scheduler is not None:
    #         self.lr_scheduler.step()

    #     return stats

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        score_indices: list[torch.LongTensor],
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`, `n`)
            score_indices (`torch.FloatTensor`)):
                Tensor of shape (`batch_size`, `n`) indicating at which response token indices to apply `scores` in rewards.
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)

        Returns:
            `torch.FloatTensor`: Per token rewards, shape (`batch_size`, `response_length`)
            `torch.FloatTensor`: Non score rewards, shape (`batch_size`, `response_length`)
            `torch.FloatTensor`: KL penalty, shape (`batch_size`, `response_length`)
        """
        rewards, non_score_rewards, kls = [], [], []

        for score, score_idxs, logprob, ref_logprob, mask in zip(scores, score_indices, logprobs, ref_logprobs, masks):
            # compute KL penalty (from difference in logprobs)
            kl = self._kl_penalty(logprob, ref_logprob)
            kls.append(kl)
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()

            # reward is preference model score + KL penalty
            # Add scores from reward model at specified indices (take off first index since logprobs start at second query token)
            for i in range(score.shape[0]):
                if (score_idxs[1:] == i).shape[0] > 0:
                    # If this reward has token indices in the score_indices, apply it at them
                    reward[score_idxs[1:] == i] += score[i]
                else:
                    # Otherwise, apply the reward at the last non-masked index (this will happen when model fails to generate parse-able questions)
                    last_non_masked_index = mask.nonzero()[-1]
                    reward[last_non_masked_index] += score

            rewards.append(reward)
        return torch.stack(rewards), torch.stack(non_score_rewards), torch.stack(kls)