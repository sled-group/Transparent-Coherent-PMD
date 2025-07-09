import numpy as np
from pprint import pprint
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import InstructBlipForConditionalGeneration
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right

def simple_lm_prompt_beam_search(lm, tokenizer, prompts, max_new_tokens=20, batch_size=20, generation_kwargs={}):
    """
    Stripped down, generic LM prompting method. This method is only tested for constrained beam search, and might not work for other generation settings.
    """
    if "num_return_sequences" in generation_kwargs:
        num_seq = generation_kwargs["num_return_sequences"]
    elif lm.generation_config.num_return_sequences:
        num_seq = lm.generation_config.num_return_sequences
    else:
        num_seq = 1

    def generate_batch(batch_prompts, generation_kwargs):
        inputs = tokenizer(text=batch_prompts, padding=True, return_tensors="pt")
        inputs = inputs.to(lm.device)

        with torch.inference_mode():
            outputs = lm.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, **generation_kwargs)

        scores = lm.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False).cpu().numpy()

        outputs = outputs.sequences[:, inputs['input_ids'].shape[-1]:]
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return outputs, scores

    all_outputs = []
    all_scores = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"running generation ({str(lm.device)})"):
        # Prepare the batch
        batch_prompts = prompts[i:i+batch_size]

        try:
            outputs, scores = generate_batch(batch_prompts, generation_kwargs)
            all_outputs += outputs
            all_scores += [round(float(np.mean(s)), 6) for s in scores] # Save sequence probability
        except Exception as e:
            pprint(e)
            print("\tWarning: Constrained beam search threw an error. Re-running batch elements one by one.")
            for prompt in batch_prompts:
                try:
                    outputs, scores = generate_batch([prompt], generation_kwargs)
                except Exception as e:
                    pprint(e)
                    print("\t\tWarning: Failed again at generating for a batch. Relaxing generation constraints.")
                    relaxed_kwargs = {k: v for k, v in generation_kwargs.items() if k not in ["constraints", "begin_suppress_tokens", "bad_words_ids"]}
                    outputs, scores = generate_batch([prompt], relaxed_kwargs)
                all_outputs += outputs
                all_scores += [round(float(np.mean(s)), 6) for s in scores] # Save sequence probability

    # Collate generated texts and scores from beam search
    all_outputs_collated = []
    all_scores_collated = []
    for beam_search_idx in range(len(all_outputs) // num_seq):
        this_outputs = []
        this_scores = []
        for i in range(num_seq):
            this_outputs.append(all_outputs[beam_search_idx * num_seq + i])
            this_scores.append(all_scores[beam_search_idx * num_seq + i])
        all_outputs_collated.append(this_outputs)
        all_scores_collated.append(this_scores)

    return all_outputs_collated, all_scores_collated

def simple_vlm_prompt_beam_search(vlm, processor, prompts, frames, image_token, max_new_tokens=20, batch_size=20, generation_kwargs={}):
    """
    Stripped down, generic LM prompting method. This method is only tested for constrained beam search, and might not work for other generation settings.
    """
    if "num_return_sequences" in generation_kwargs:
        num_seq = generation_kwargs["num_return_sequences"]
    elif vlm.generation_config.num_return_sequences:
        num_seq = vlm.generation_config.num_return_sequences
    else:
        num_seq = 1

    all_outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"running generation ({str(vlm.device)})"):
        # Prepare the batch
        batch_prompts = prompts[i:i+batch_size]
        batch_frames = frames[i:i+batch_size]

        inputs = processor(text=batch_prompts, images=batch_frames, padding=True, return_tensors="pt")
        inputs = inputs.to(vlm.device)

        with torch.inference_mode():
            outputs = vlm.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, **generation_kwargs)

        outputs = outputs.sequences[:, inputs['input_ids'].shape[-1]:]
        outputs = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        all_outputs += outputs

    # Collate generated texts and scores from beam search
    all_outputs_collated = []
    for beam_search_idx in range(len(all_outputs) // num_seq):
        this_outputs = []
        for i in range(num_seq):
            this_outputs.append(all_outputs[beam_search_idx * num_seq + i])
        all_outputs_collated.append(this_outputs)

    return all_outputs_collated


def simple_lm_prompt(lm, tokenizer, prompts, max_new_tokens=20, batch_size=20, generation_kwargs={}):
    """
    Stripped down, generic LM prompting method. This method is only tested for constrained beam search, and might not work for other generation settings.
    """
    all_outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"running generation ({str(lm.device)})"):
        # Prepare the batch
        batch_prompts = prompts[i:i+batch_size]

        inputs = tokenizer(text=batch_prompts, padding=True, return_tensors="pt")
        inputs = inputs.to(lm.device)

        outputs = lm.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, **generation_kwargs)

        outputs = outputs.sequences[:, inputs['input_ids'].shape[-1]:]
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        all_outputs += outputs

    return all_outputs

# Below is not in use and not debugged
# def simple_prompt_vlm(vlm, vlm_processor, frames, prompts, max_new_tokens=20, generation_kwargs={}):
   
#     inputs = vlm_processor(text=prompts, images=frames, padding=True, return_tensors="pt")
#     inputs = inputs.to(vlm.device)

#     outputs = vlm.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, **generation_kwargs)
#     outputs = vlm_processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    
#     if type(vlm) == LlavaForConditionalGeneration:
#         outputs = [[output.replace("USER:  ", "USER: <image>") for output in beam_search_outputs] for beam_search_outputs in outputs]
#         outputs = [[output.replace(prompt, "") for output in beam_search_outputs] for beam_search_outputs, prompt in zip(outputs, prompts)]
#     else:
#         raise NotImplementedError(f"simple_prompt doesn't support VLM type {type(vlm)}!")
    
#     return outputs

def compute_completion_log_likelihoods(model, tokenizer, prompts: list[str], completions: list[list[str]], batch_size: int):
    # Tokenize prompts
    tokenized_prompts = [tokenizer(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'][0] for prompt in prompts]
    
    # Tokenize completions
    tokenized_completions = [[tokenizer(completion, return_tensors='pt', add_special_tokens=True)['input_ids'][0] for completion in completion_list] for completion_list in completions]
    
    # Find the maximum length of prompts and completions
    max_prompt_length = max(len(prompt) for prompt in tokenized_prompts)
    max_completion_length = max(len(completion) for completion_list in tokenized_completions for completion in completion_list)
    
    # Pad prompts
    padded_prompts = torch.stack([torch.cat((torch.tensor([tokenizer.pad_token_id] * (max_prompt_length - len(prompt))).long().to(model.device), prompt.to(model.device)), dim=-1) for prompt in tokenized_prompts])

    # Flatten and pad completions
    flattened_completions = [completion for completion_list in tokenized_completions for completion in completion_list]
    padded_completions = torch.stack([torch.cat((completion.to(model.device), torch.tensor([tokenizer.pad_token_id] * (max_completion_length - len(completion))).long().to(model.device)), dim=-1) for completion in flattened_completions])
    
    # Prepare combined input
    combined_inputs = []
    prompt_completion_mapping = []
    
    idx = 0
    for prompt_idx, prompt in enumerate(padded_prompts):
        for completion_idx in range(len(tokenized_completions[prompt_idx])):
            combined_input = torch.cat((prompt, padded_completions[idx]), dim=-1)
            combined_inputs.append(combined_input)
            prompt_completion_mapping.append((prompt_idx, completion_idx))
            idx += 1
    
    combined_inputs = torch.stack(combined_inputs).to(model.device)
    
    # Split combined inputs into batches
    num_batches = (combined_inputs.size(0) + batch_size - 1) // batch_size
    all_log_likelihoods = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, combined_inputs.size(0))
        batch_inputs = combined_inputs[batch_start:batch_end]

        # Get the logits from the model in a single forward pass
        with torch.inference_mode():
            outputs = model(batch_inputs)

        logits = outputs.logits[:, :-1, :]  # Exclude the last token's logits
        sequences = batch_inputs[:, 1:].contiguous()  # Shift input sequences for alignment

        # Compute log softmax to get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather the log probabilities of the selected tokens
        selected_log_probs = torch.gather(log_probs, 2, sequences.unsqueeze(-1)).squeeze(-1)
        
        # Calculate the average log likelihood for each completion in the batch
        for i in range(batch_inputs.size(0)):
            completion_start_index = max_prompt_length
            completion_end_index = completion_start_index + max_completion_length
            non_pad_mask = (sequences[i, completion_start_index:completion_end_index] != tokenizer.pad_token_id)
            completion_log_probs = selected_log_probs[i, completion_start_index:completion_end_index]
            non_pad_log_probs = completion_log_probs[non_pad_mask]
            average_log_likelihood = non_pad_log_probs.mean().item()
            all_log_likelihoods.append(average_log_likelihood)
    
    # Organize all_log_likelihoods into a list of lists corresponding to the prompts and their completions
    results = [[] for _ in range(len(prompts))]
    for (prompt_idx, completion_idx), log_likelihood in zip(prompt_completion_mapping, all_log_likelihoods):
        results[prompt_idx].append(log_likelihood)
    
    return results
    
import torch

def compute_completion_log_likelihoods_encoder_decoder(model, tokenizer, prompts: list[str], completions: list[list[str]], batch_size: int):
    # Tokenize prompts
    tokenized_prompts = [tokenizer(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'][0] for prompt in prompts]
    
    # Tokenize completions
    completions = [[f"{tokenizer.pad_token} {completion}" for completion in some_completions] for some_completions in completions] # NOTE: pad token signals to InstructBLIP encoder-decoder model to start - but this isn't compatible with all encoder-decoder models
    tokenized_completions = [[tokenizer(completion, return_tensors='pt', add_special_tokens=True)['input_ids'][0] for completion in completion_list] for completion_list in completions]
    
    # Find the maximum length of prompts and completions
    max_prompt_length = max(len(prompt) for prompt in tokenized_prompts)
    max_completion_length = max(len(completion) for completion_list in tokenized_completions for completion in completion_list)
    
    # Pad prompts
    padded_prompts = torch.stack([torch.cat((torch.tensor([tokenizer.pad_token_id] * (max_prompt_length - len(prompt))).long().to(model.device), prompt.to(model.device)), dim=-1) for prompt in tokenized_prompts])

    # Flatten and pad completions
    flattened_completions = [completion for completion_list in tokenized_completions for completion in completion_list]
    padded_completions = torch.stack([torch.cat((completion.to(model.device), torch.tensor([tokenizer.pad_token_id] * (max_completion_length - len(completion))).long().to(model.device)), dim=-1) for completion in flattened_completions])
    
    # Split into batches
    num_batches = (len(flattened_completions) + batch_size - 1) // batch_size
    all_log_likelihoods = []

    idx = 0
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(flattened_completions))

        batch_prompts = torch.cat([padded_prompts[i].unsqueeze(0).repeat(len(completions[i]), 1) for i in range(len(prompts))], dim=0)[batch_start:batch_end]
        batch_completions = padded_completions[batch_start:batch_end]

        # Run forward pass
        with torch.inference_mode():
            outputs = model(input_ids=batch_prompts, decoder_input_ids=batch_completions)
        
        logits = outputs.logits[:, :-1, :]  # Exclude the last token's logits
        sequences = batch_completions[:, 1:].contiguous()  # Shift input sequences for alignment

        # Compute log softmax to get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather the log probabilities of the selected tokens
        selected_log_probs = torch.gather(log_probs, 2, sequences.unsqueeze(-1)).squeeze(-1)
        
        # Calculate the average log likelihood for each completion in the batch
        for i in range(batch_end - batch_start):
            non_pad_mask = (sequences[i] != tokenizer.pad_token_id)
            completion_log_probs = selected_log_probs[i]
            non_pad_log_probs = completion_log_probs[non_pad_mask]
            average_log_likelihood = non_pad_log_probs.mean().item()
            all_log_likelihoods.append(average_log_likelihood)
    
    # Organize all_log_likelihoods into a list of lists corresponding to the prompts and their completions
    results = [[] for _ in range(len(prompts))]
    idx = 0
    for prompt_idx, completion_list in enumerate(completions):
        for completion_idx in range(len(completion_list)):
            results[prompt_idx].append(all_log_likelihoods[idx])
            idx += 1
    
    return results


def compute_completion_log_likelihoods_vlm(model, processor, prompts: list[str], frames: list[str], completions: list[list[str]], batch_size: int):
    assert len(prompts) == len(frames)
    
    tokenizer = processor.tokenizer
    
    # Hold a pointer to frame for each completion
    frames = [frame for frame, completions in zip(frames, completions) for _ in completions]

    # Tokenize prompts
    tokenized_prompts = [tokenizer(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'][0] for prompt in prompts]
    
    # Tokenize completions
    tokenized_completions = [[tokenizer(completion, return_tensors='pt', add_special_tokens=True)['input_ids'][0] for completion in completion_list] for completion_list in completions]
    
    # Find the maximum length of prompts and completions
    max_prompt_length = max(len(prompt) for prompt in tokenized_prompts)
    max_completion_length = max(len(completion) for completion_list in tokenized_completions for completion in completion_list)
    
    # Pad prompts
    padded_prompts = torch.stack([torch.cat((torch.tensor([tokenizer.pad_token_id] * (max_prompt_length - len(prompt))).long().to(model.device), prompt.to(model.device)), dim=-1) for prompt in tokenized_prompts])

    # Flatten and pad completions
    flattened_completions = [completion for completion_list in tokenized_completions for completion in completion_list]
    padded_completions = torch.stack([torch.cat((completion.to(model.device), torch.tensor([tokenizer.pad_token_id] * (max_completion_length - len(completion))).long().to(model.device)), dim=-1) for completion in flattened_completions])
    
    # Prepare combined input
    combined_inputs = []
    prompt_completion_mapping = []
    
    idx = 0
    for prompt_idx, prompt in enumerate(padded_prompts):
        for completion_idx in range(len(tokenized_completions[prompt_idx])):
            combined_input = torch.cat((prompt, padded_completions[idx]), dim=-1)
            combined_inputs.append(combined_input)
            prompt_completion_mapping.append((prompt_idx, completion_idx))
            idx += 1
    
    combined_inputs = torch.stack(combined_inputs).to(model.device)
    
    # Split combined inputs into batches
    num_batches = (combined_inputs.size(0) + batch_size - 1) // batch_size
    all_log_likelihoods = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, combined_inputs.size(0))
        batch_inputs = combined_inputs[batch_start:batch_end]

        batch_frames = frames[batch_start:batch_end]
        image_inputs = processor.image_processor(batch_frames, return_tensors="pt")

        attention_mask = torch.ones_like(batch_inputs)
        attention_mask[batch_inputs == tokenizer.pad_token_id] = 0.0

        # Get the logits from the model in a single forward pass
        with torch.inference_mode():
            outputs = model(batch_inputs, attention_mask=attention_mask, **image_inputs)
        

        logits = outputs.logits[:, :-1, :]  # Exclude the last token's logits
        sequences = batch_inputs[:, 1:].contiguous()  # Shift input sequences for alignment

        # Compute log softmax to get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather the log probabilities of the selected tokens
        selected_log_probs = torch.gather(log_probs, 2, sequences.unsqueeze(-1)).squeeze(-1)
        
        # Calculate the average log likelihood for each completion in the batch
        for i in range(batch_inputs.size(0)):
            completion_start_index = max_prompt_length
            completion_end_index = completion_start_index + max_completion_length
            non_pad_mask = (sequences[i, completion_start_index:completion_end_index] != tokenizer.pad_token_id)
            completion_log_probs = selected_log_probs[i, completion_start_index:completion_end_index]
            non_pad_log_probs = completion_log_probs[non_pad_mask]
            average_log_likelihood = non_pad_log_probs.mean().item()
            all_log_likelihoods.append(average_log_likelihood)
    
    # Organize all_log_likelihoods into a list of lists corresponding to the prompts and their completions
    results = [[] for _ in range(len(prompts))]
    for (prompt_idx, completion_idx), log_likelihood in zip(prompt_completion_mapping, all_log_likelihoods):
        results[prompt_idx].append(log_likelihood)
    
    return results
    

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# repr_cache = {}
def get_embeddings(model, tokenizer, texts, batch_size=16):
    """Gets sentence embeddings given a sentence transformer model and tokenizer."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[i:i+batch_size], padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            # Use [CLS] token or mean pooling
            pooled = mean_pooling(outputs, inputs['attention_mask'])
            embeddings.append(pooled.cpu())
        embeddings = torch.stack(embeddings).squeeze(0)
        # embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings