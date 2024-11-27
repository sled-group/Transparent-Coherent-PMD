import os
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from typing import Callable, Optional
import base64
import pickle
import torch
import numpy as np
from openai import AzureOpenAI, OpenAI

from travel.data.mistake_detection import MistakeDetectionDataset, MistakeDetectionExample
from travel.data.vqa import VQAOutputs, VQAResponse
from travel.data.vqg import VQGInputs, VQGOutputs, parse_vqg_outputs, save_vqg_outputs
from travel.model.grounding import VisualFilterTypes, AdaptiveVisualFilter
from travel.data.utils.image import resize_with_aspect, CACHED_FRAME_DIMENSION
from travel.model.mistake_detection import DETECTION_FRAMES_PROPORTION
from travel.constants import CACHE_FREQUENCY
from travel.model.nli import NLI_MODEL_PATH, run_nli, NLI_HYPOTHESIS_TEMPLATE
from travel.model.metrics import entropy_tensor

class GPT:
    def __init__(self, api_key, endpoint, model_name) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.model_name = model_name
    
    def get_vqa_response_token_ids(self):
        if self.model_name[:6] == 'gpt-4o':
            responses = {VQAResponse.No: 3160, VQAResponse.Yes: 13022}
        else:
            responses = {VQAResponse.No: 2822, VQAResponse.Yes: 9642}
        return responses
    
    def _encode_image(self, frame: Image.Image) -> str:
        """
        Encodes a PIL.Image.Image object to a base64 string.

        :param image_path: A PIL.Image.Image object to encode.
        """
        buffered = BytesIO()
        image = frame.convert('RGB') # Discard alpha channel if needed
        image.save(buffered, format="JPEG")  # You can change the format if necessary
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return encoded_string
    
    def _example_in_outputs(self, example_id: str, vqa_outputs: list[VQAOutputs]) -> bool:
        """
        Helper function to check if an example_id appears in any of the vqa_outputs.

        :param example_id: The example_id to look for.
        :param vqa_outputs: List of VQAOutputs to check.

        """
        for output in vqa_outputs:
            if output[0][0].example_id == example_id:
                return True
        return False
    
    def _is_yes(self, token: str) -> bool:
        """
        Returns True if the given token is 'yes', 'Yes, ' yes', or ' Yes'. False otherwise.

        :param token: A token from the API response.
        """
        if (token == 'yes' or token == 'Yes'
            or token == ' yes' or token == ' Yes'):
            return True
        return False

    def _is_no(self, token: str) -> bool:
        """
        Returns True if the given token is 'no', 'No, ' np', or ' No'. False otherwise.

        :param token: A token from the API response.
        """
        if (token == 'no' or token == 'No'
            or token == ' no' or token == ' No'):
            return True
        return False
    
    def _get_probs(self, gpt_response) -> dict[VQAResponse, float]:
        """
        Returns the normalized porbabilities of 'yes' answer and 'no' answer using the given API response.
        
        :param  gpt_response: Response from the API containing top log probabilities of each token.
        """
        logprobs = gpt_response.choices[0].logprobs.content

        yes_prob = None
        no_prob = None

        # Find the first token that is a 'yes' or 'no' and treat it as the model's answer.
        # Grab the corresponding top log porbabilties of that token to get the porbabilities of 'yes' and 'no'.
        for entry in logprobs:
            token = entry.token
            top_probs = entry.top_logprobs
            if self._is_yes(token) or self._is_no(token):
                for prob in top_probs:
                    if self._is_yes(prob.token) and yes_prob is None:
                        yes_prob = prob.logprob
                    elif self._is_no(prob.token) and no_prob is None:
                        no_prob = prob.logprob
                break

        # If one doesn't appear set its porbability to 0 and the other to 1.
        # If both don't appear, set both to 0.5.
        # Normalize if both appear
        if yes_prob == None and no_prob == None:
            yes_prob = no_prob = 0.5
        elif yes_prob == None:
            yes_prob = 0
            no_prob = 1
        elif no_prob == None:
            no_prob = 0
            yes_prob = 1
        else:
            yes_prob = np.exp(yes_prob)
            no_prob = np.exp(no_prob)
            old_yes_prob = yes_prob
            yes_prob = old_yes_prob / (old_yes_prob + no_prob)
            no_prob = no_prob / (old_yes_prob + no_prob)
        
        return {VQAResponse.No: no_prob, VQAResponse.Yes: yes_prob}
        

    def prompt_gpt(self,
                    prompt_text: str,
                    image: Image=None,
                    temperature: float=0.0,
                    max_tokens: int=128,
                    logprobs: bool=False,
                    top_logprobs: int=0):
        """
        Sends a request to the GPT API and returns the response object.

        :param prompt_text: Text prompt for GPT.
        :param Image: Optional image to include in the prompt.
        :param temperature: Sampling temperature used for the GPT model. 
        :param max_tokens: Maximum number of tokens to be generated by GPT.
        :param logprobs: Boolean indicating if to include the log probabilities of every output token.
        :param top_logprobs: The number of top probabilities to return for every output token, in [0,20]. logprobs has to be True for this option.
        """

        client = AzureOpenAI(
            api_key=self.api_key,  
            api_version="2024-03-01-preview",
            azure_endpoint=self.endpoint,
        )
        if image is not None:
            encoded_image = self._encode_image(image)
            if not logprobs:
                top_logprobs = None
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs
            )
        else:
            if not logprobs:
                top_logprobs = None
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", "text": prompt_text
                        }
                    ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                top_logprobs=top_logprobs
            )
        return response
    

    def run_vqa(self,
                eval_dataset: MistakeDetectionDataset,
                generate_prompts: Callable[[MistakeDetectionExample], tuple[list[str], list[str], list[VQAResponse], list[Image.Image]]],
                cache_dir: str,
                cache_frequency: int=CACHE_FREQUENCY,
                temperature: float=0,
                ) -> list[list[list[VQAOutputs]]]:
        """
        Method to run VQA on a MistakeDetectionDataset using calls to the GPT-4 API.
        
        :param eval_dataset: MistakeDetectionDataset to run inference on.
        :param generate_prompts: A method that generates a list of prompts from a single MistakeDetectionExample.
        :param cache_dir: Directory to cache outputs in.
        :param cache_frequency: Determines how frequent VQAOutputs are cached.
        :param temprature: Sampling temperature used for the GPT model.
        """
        vqa_outputs = []

        # Check and load if there are cached outputs 
        cache_fname = os.path.join(cache_dir, "cached_outputs.pkl")
        if os.path.exists(cache_fname):
            with open(cache_fname, 'rb') as f:
                vqa_outputs = pickle.load(f)

        last_save = len(vqa_outputs)
        # Iterate through dataset, skip over loaded cached output
        for ex in tqdm(eval_dataset.get_batches(batch_size=1),
                            desc=f"VQA using GPT API", 
                            total=len(eval_dataset)):
            example = ex
            # Skip if the outputs for this example were already loaded from the cache
            if self._example_in_outputs(example.example_id, vqa_outputs):
                continue
            
            example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION)
            # Prompts for this example
            questions, prompts, answers, frames = generate_prompts(example)
            assert len(questions) == len(prompts) == len(answers) == len(frames), "Passed `generate_prompts` method must return same number of questions, prompts, answers, and frames!"
            
            # Send API Requests for every prompt
            example_outputs = []
            for idx, prompt in enumerate(prompts):
                response = self.prompt_gpt(prompt_text=prompt,
                                        image=frames[idx],
                                        temperature=temperature,
                                        logprobs=True,
                                        top_logprobs=20)
                answer_probs = self._get_probs(response)
                example_outputs.append(
                    [VQAOutputs(
                        example.task_name,
                        example.example_id,
                        example.procedure_id,
                        frames[idx],
                        prompt, 
                        answers[idx],
                        {},
                        None,
                        questions[idx],
                        answer_probs=answer_probs
                    )]
                )
            vqa_outputs.append(example_outputs)
            # Cache based on frequency
            if len(vqa_outputs) - last_save >= cache_frequency:
                with open(cache_fname, 'wb') as f:
                    pickle.dump(vqa_outputs, f)
                last_save = len(vqa_outputs)
        return vqa_outputs 
    

    def run_vqg(self,
                inputs: list[VQGInputs],
                input_ids: list[str],
                temperature: float=1,
                top_p: float=1,
                save_path: Optional[str]=None,
                vqg_outputs: dict[str, VQGOutputs]={},
                omit_failed_instances: bool=True) -> dict[str, Optional[VQGOutputs]]:
        """
        Runs VQG with a given LM text generation pipeline and list of VQG inputs.

        :param inputs: List of VQG inputs, including procedures, prompts, etc.
        :param input_ids: Unique string identifiers for inputs. These may characterize a specific prompt or run of a prompt (e.g., at a different temperature).
        :param temprature: Sampling temperature used for the GPT model.
        :param save_path: Optional path to save VQG outputs during and after running. Must either be a json filename or path to a directory.
        :param vqg_outputs: Partly filled dictionary of VQG outputs to start from; only pass this if starting from a partially completed run of VQG, and make sure complete/incomplete prompts are managed appropriately outside of this method.
        :return: Completed dictionary of VQGOutputs.
        """
        assert len(inputs) == len(input_ids), "run_vqg expected the same number of inputs and input IDs!"
        prompt_idx = 0
        for input, input_id in tqdm(zip(inputs, input_ids),
                                    desc="running VQG (GPT)",
                                    total=len(inputs)):
            
            procedure_id = int(input.procedure_id)
            step = input.procedure_description
            response = self.prompt_gpt(prompt_text=input.prompt,
                                    temperature=temperature,
                                    top_p=top_p)
            text = response.choices[0].message.content
            # Parse reported target object and questions and answers
            try:
                output = parse_vqg_outputs(text, procedure_id, step)
            except:
                print("Warning: failed to parse a VQG output.")
                print(text)

                if not omit_failed_instances:
                    vqg_outputs[input_id] = None

                continue

            vqg_outputs[input_id] = output
            if prompt_idx % CACHE_FREQUENCY == 0 and save_path is not None:
                print("Saving progress...")
                save_vqg_outputs(vqg_outputs, save_path)

            prompt_idx += 1

        # Save progress one last time after completion
        save_vqg_outputs(vqg_outputs, save_path)
        return vqg_outputs
    
    def generate_questions(self,
                           prompts: list[str],
                           max_tokens: int=20,
                           temperature: float=0.0):
        """
        Generates questions with the given prompts using GPT. 
        """
        all_questions = []
        for prompt in prompts:
            api_response = self.prompt_gpt(prompt_text=prompt,
                                           temperature=temperature,
                                           max_tokens=max_tokens)
            question = api_response.choices[0].message.content
            all_questions.append(question)
        return all_questions
    
    def run_GPT_vqa(self,
                    prompts: list[str],
                    frames: list[Image.Image],
                    temperature: float=0.0) -> list:
        """
        Runs VQA for given prompts and frames in batches with a given VLM and its processor.

        :param vlm: VLM for conditional generation from `transformers`.
        :param process: VLM processor from `transformers`, including tokenizer and image processor.
        :param prompts: List of prompts including visual questions.
        :param frames: List of images to ask visual questions about.
        :param batch_size: Batch size for running inference.
        :param cache_path: .pt file to cache incomplete logits in.
        :param return_attention: Whether to return attentions for passed prompts in addition to logits.
        :return: Full tensor of logits output from each question. The process of mapping this into VQAOutputs instances requires task/process-specific information, so it should be done outside of this method.
        """
        assert len(prompts) == len(frames), "Need same number of prompts and frames to run VQA!"

        all_probs = []
        for prompt, frame in zip(prompts, frames):
            response = self.prompt_gpt(prompt_text=prompt,
                                       image=frame,
                                       temperature=temperature,
                                       logprobs=True,
                                       top_logprobs=20)
            answer_probs = self._get_probs(response)
            all_probs.append(answer_probs)
        return all_probs


    
    def run_GPT_vqa_with_visual_filter(self, batch_examples, batch_frames, prompts_a, new_questions, question_idx, batch_size, visual_filter=None, nlp=None, visual_filter_mode=None, frame_cache_dir=None, is_encoder_decoder=False):
        """
        VQA and visual filter wrapper method for iterative VQA experiments.

        :param vlm_processor: VLM processor.
        :param vlm: VLM.
        :param batch_examples: Batch of MistakeDetectionExample.
        :param batch_frames: Batch of frames (PIL images).
        :param prompts_a: Full string prompts to get a yes/no answer.
        :param new_questions: Last generated questions to use with text-conditioned visual filters.
        :param question_idx: Index or identifier of the current question. This is only used to save modified frames from the visual filter.
        :param batch_size: Batch size for VQA.
        :param visual_filter: Optional training free visual filter to modify images and possibly run VQA twice.
        :param nlp: spaCy NLP pipeline.
        :param visual_filter_mode: Type of visual filter.
        :param frame_cache_dir: Directory to cache visual filter modified frames for later inspection. If not passed, will not save any frames.
        :return: Logits from VQA.
        """
        # Apply visual filter to frames for VQA
        if visual_filter:
            if visual_filter_mode == VisualFilterTypes.Contrastive_Region:
                batch_frames_filtered = visual_filter(nlp, batch_frames, new_questions)
            elif visual_filter_mode == VisualFilterTypes.Visual_Contrastive:
                batch_frames_filtered = visual_filter(batch_frames)
            elif visual_filter_mode in [VisualFilterTypes.Spatial_NoRephrase, VisualFilterTypes.Spatial_Blur]:
                batch_frames_filtered, _ = visual_filter(nlp, batch_frames, new_questions, return_visible_target_objects=False)
            elif visual_filter_mode == VisualFilterTypes.Spatial:
                batch_frames_filtered, new_questions = visual_filter(nlp, batch_frames, new_questions, return_visible_target_objects=False)
            elif visual_filter_mode == VisualFilterTypes.AGLA:
                batch_frames_filtered = visual_filter(batch_frames, new_questions)

        # Cache paths to frames (if using a visual filter, save filtered frames and cache paths to them)
        if not(visual_filter is None or frame_cache_dir is None):
            for batch_sub_idx, (frame, example) in enumerate(zip(batch_frames_filtered, batch_examples)):
                this_frame_cache_dir = os.path.join(frame_cache_dir, f"vqa_frames/{example.example_id}")
                if not os.path.exists(this_frame_cache_dir):
                    os.makedirs(this_frame_cache_dir)
                frame_path = os.path.join(this_frame_cache_dir, f"frame_q{question_idx}.jpg")
                resized_frame = resize_with_aspect(frame, CACHED_FRAME_DIMENSION)
                resized_frame.save(frame_path)

        # Run VQA on base image (yes/no)
        if not (visual_filter and visual_filter_mode in [VisualFilterTypes.Spatial_NoRephrase, VisualFilterTypes.Spatial_Blur]):
            new_answers_logits = self.run_GPT_vqa(prompts_a, batch_frames)
        else:
            # Spatial filter doesn't need original image logits, so don't get them for efficiency
            new_answers_logits = None

        # Run VQA on filtered image if needed and combine logits as proposed in approaches' papers
        if visual_filter:
            new_answers_logits_filtered = self.run_GPT_vqa(prompts_a, batch_frames_filtered)
            # new_answers_logits = visual_filter.combine_logits(new_answers_logits, new_answers_logits_filtered)

        return new_answers_logits

    def rephrase_question_answer_GPT(self, questions: list[str], answers: list[str], temperature: float=0, max_tokens: int=20):
        # return f"{question} {answer.name}."
        assert all(a == "Yes" or a == "No" for a in answers), "Expected all answers to be 'Yes' or 'No', but got " + str(answers)
        examples = [
            "Question: Is there a bowl on the table?\nAnswer: Yes\nStatement: There is a bowl on the table.",
            "Question: Are the eggs cracked?\nAnswer: No\nStatement: The eggs are not cracked.",
            "Question: Does the cardboard box look open?\nAnswer: Yes\nStatement: The cardboard box looks open.",
            "Question: Are there any leaves outside of the basket?\nAnswer: No\nStatement: There are not any leaves outside of the basket.",
            "Question: Is the orange peeled?\nAnswer: Yes\nStatement: The orange is peeled.",
            "Question: Is the mug empty?\nAnswer: No\nStatement: The mug is not empty.",
            "Question: Are there hedge trimmers in the image?\nAnswer: Yes\nStatement: There are hedge trimmers in the image.",
            "Question: Has the light switch been turned on?\nAnswer: No\nStatement: The light switch has not been turned on.",
            "Question: Does the table have any cups on it?\nAnswer: Yes\nStatement: The table has cups on it.",
            "Question: Is the cabinet closed?\nAnswer: No\nStatement: The cabinet is not closed.",
        ]
        prompts = ["\n\n".join(examples) + f"\n\nQuestion: {question}\nAnswer: {answer}\nStatement: " for question, answer in zip(questions, answers)]
        rephrased_texts = []
        for prompt in prompts:
            api_response = self.prompt_gpt(prompt_text=prompt,
                                        temperature=temperature,
                                        max_tokens=max_tokens)
            text = api_response.choices[0].message.content
            rephrased_texts.append(text) 
        rephrased_texts = [text.split(".")[0] + "." for text in rephrased_texts]
        rephrased_texts = [text.strip() for text in rephrased_texts]
        return rephrased_texts

    def rephrase_procedure_success_GPT(self, procedures: list[str], temperature: float=0, max_tokens: int=20):
        examples = [
            "Procedure: Soak the sponge in a soapy water with your hands.\nStatement: The sponge has been successfully soaked in soapy water with someone's hands.",
            "Procedure: Turn on a torch light.\nStatement: The torch light has been successfully turned on.",
            "Procedure: Fold the right edge of the wrapper.\nStatement: The right edge of the wrapper has been successfully folded.",
            "Procedure: Pour the water into the blue container.\nStatement: The water has been successfully poured into the blue container.",
            "Procedure: Spread the black peas on the salad with the spoon in your hand.\nStatement: The black peas have been successfully spread on the salad with the spoon in someone's hand.",
            "Procedure: Pick the scrubber from the sink.\nStatement: The scrubber has been successfully picked from the sink.",
            "Procedure: Peel the onion.\nStatement: The onion has been successfully peeled.",
            "Procedure: Put the dirt in the dust bin.\nStatement: The dirt has been successfully put in the dust bin.",
            "Procedure: Cut dough in two.\nStatement: The dough has been successfully cut in two.",
            "Procedure: Close the fridge.\nStatement: The fridge has been successfully closed.",
        ]
        prompts = ["\n\n".join(examples) + f"\n\Procedure: {procedure}\nStatement: " for procedure in procedures]
        rephrased_texts = []
        for prompt in prompts:
            api_response = self.prompt_gpt(prompt_text=prompt,
                                        temperature=temperature,
                                        max_tokens=max_tokens)
            text = api_response.choices[0].message.content
            rephrased_texts.append(text) 
        rephrased_texts = [text.split(".")[0] + "." for text in rephrased_texts]
        rephrased_texts = [text.strip() for text in rephrased_texts]
        print(procedures)
        print(rephrased_texts)
        print("===================")
        return rephrased_texts

    def question_coherence_metrics_nli_GPT(self, nli_tokenizer, nli_model, procedures: list[str], questions: list[str], 
                                    answers: Optional[list[str]]=None, 
                                    previous_questions: Optional[list[list[str]]]=None, 
                                    previous_answers: Optional[list[list[str]]]=None, 
                                    mistake_labels: Optional[list[bool]]=None,
                                    temperature: float=0.0,
                                    max_tokens: int=20, 
                                    rephrase_success=False):
        """
        Calculates coherence metrics for candidate questions about procedures in iterative VQA.
        """
        if answers is not None:
            assert all(a in ["Yes", "No"] for a in answers)
        if previous_answers is not None:
            assert all(a in ["Yes", "No"] for aa in previous_answers for a in aa)
        
        metrics = {}
        
        if not rephrase_success:
            hypothesis_procedure = [NLI_HYPOTHESIS_TEMPLATE.format(procedure=procedure) for procedure in procedures]
        else:
            hypothesis_procedure = self.rephrase_procedure_success_GPT(procedures, temperature, max_tokens)

        # Rephrase question with a yes and no answer as statements to compare their entailment probability of success
        rephrased_yes = self.rephrase_question_answer_GPT(questions, ["Yes"] * len(questions), temperature, max_tokens)
        rephrased_no = self.rephrase_question_answer_GPT(questions, ["No"] * len(questions), temperature, max_tokens)    
        metrics['rephrased_questions_yes'] = rephrased_yes
        metrics['rephrased_questions_no'] = rephrased_no
        premise_yes = rephrased_yes
        premise_no = rephrased_no
        probs_yes = run_nli(nli_tokenizer, nli_model, list(zip(premise_yes, hypothesis_procedure)))
        probs_no = run_nli(nli_tokenizer, nli_model, list(zip(premise_no, hypothesis_procedure)))
        if answers:
            probs_actual = torch.stack([probs_yes[i] if answers[i] == "Yes" else probs_no[i] for i in range(len(answers))])

        # Individual relevance: how much probability of success changes depending on the answer
        relevance = torch.abs(probs_yes[:, 0] - probs_no[:, 0])
        metrics['relevance'] = relevance.numpy()

        # Potential informativeness: at most (or for actual answer), how confident will we be that the answer to the question would indicate a success or mistake
        if not answers:
            informativeness_yes = 1.0 - entropy_tensor(probs_yes[:, 0])
            informativeness_no = 1.0 - entropy_tensor(probs_no[:, 0])
            informativeness = torch.max(torch.cat((informativeness_yes.unsqueeze(1), informativeness_no.unsqueeze(1)), dim=-1), dim=-1).values
        else:
            informativeness = 1.0 - entropy_tensor(probs_actual[:, 0])
        metrics['informativeness'] = informativeness.numpy()

        if previous_questions:
            assert len(previous_questions) == len(previous_answers), "Expected same number of questions and answers!"

            # Flatten and rephrase past questions and answers into statements, then un-flatten
            rephrased_past = self.rephrase_question_answer_GPT(
                [question for p_questions in previous_questions for question in p_questions],
                [answer for p_answers in previous_answers for answer in p_answers],
                temperature,
                max_tokens
            )
            parallel_idx = 0
            new_rephrased_past = []
            for p_questions in previous_questions:
                this_rephrased_past = []
                for _ in p_questions:
                    this_rephrased_past.append(rephrased_past[parallel_idx])
                    parallel_idx += 1
                new_rephrased_past.append(this_rephrased_past)
            rephrased_past = new_rephrased_past

            premise_past = [" ".join(past_qs_rephrased) for past_qs_rephrased in rephrased_past]
            
            premise_past_yes = [(pp + " " + py).strip() for pp, py in zip(premise_past, premise_yes)]
            premise_past_no = [(pp + " " + pn).strip() for pp, pn in zip(premise_past, premise_no)]
            # probs_past = run_nli(nli_tokenizer, nli_model, list(zip(premise_past, hypothesis_procedure)))
            probs_past_yes = run_nli(nli_tokenizer, nli_model, list(zip(premise_past_yes, hypothesis_procedure)))
            probs_past_no = run_nli(nli_tokenizer, nli_model, list(zip(premise_past_no, hypothesis_procedure)))
            if answers:
                probs_past_actual = torch.stack([probs_past_yes[i] if answers[i] == "Yes" else probs_past_no[i] for i in range(len(answers))])
            
            # Marginal relevance: how much probability of success changes depending on the answer AND information we already extracted from the image
            relevance_marginal = torch.abs(probs_past_yes[:, 0] - probs_past_no[:, 0])
            metrics['relevance_marginal'] = relevance_marginal.numpy()

            # Marginal expected informativeness: with past questions and answers, how informative could (or is) the answer to this question be toward the final decision
            if not answers:
                informativeness_yes = 1.0 - entropy_tensor(probs_past_yes[:, 0])
                informativeness_no = 1.0 - entropy_tensor(probs_past_no[:, 0])
                informativeness_marginal = torch.max(torch.cat((informativeness_yes.unsqueeze(1), informativeness_no.unsqueeze(1)), dim=-1), dim=-1).values
            else:
                informativeness_marginal = 1.0 - entropy_tensor(probs_past_actual[:, 0])
            metrics['informativeness_marginal'] = informativeness_marginal.numpy()
        else:
            metrics['relevance_marginal'] = metrics['relevance']
            metrics['informativeness_marginal'] = metrics['informativeness']

        # "Verifiability" metrics weight marginal informativeness by marginal relevance
        metrics['informativeness_x_relevance_marginal'] = metrics['informativeness'] * metrics['relevance_marginal']
        metrics['informativeness_marginal_x_relevance_marginal'] = metrics['informativeness_marginal'] * metrics['relevance_marginal']

        if answers is not None and mistake_labels is not None:
            # Calculate an alternative (not reference free) form of informativeness that is negative if leaning toward the incorrect final answer (for mistake or success)
            # (this can only be done when answers is provided, which is during final coherence evaluation rather than coherence-based reranking)
            leaning_toward_mistake = torch.tensor([1 if p < 0.5 else -1 for p in probs_past_actual[:, 1]])
            actually_is_mistake = torch.tensor([1 if l else -1 for l in mistake_labels])
            multipliers = leaning_toward_mistake * actually_is_mistake # This will be 1 if leaning the correct way, otherwise -1
            assert sum(multipliers.shape) == len(mistake_labels)
            multipliers = multipliers.numpy()

            for k in ['informativeness', 'informativeness_x_relevance_marginal', 'informativeness_marginal', 'informativeness_marginal_x_relevance_marginal']:
                metrics[k + "_ref"] = metrics[k] * multipliers

        # Convert to floats to ensure json serializable
        metrics = {
            k: [round(float(val), 6) for val in v] if type(v[0]) != str else v
            for k, v in metrics.items()
        }
        return metrics



        