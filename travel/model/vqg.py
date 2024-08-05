import numpy as np
import torch
from tqdm import tqdm
from transformers import TextGenerationPipeline
from transformers.pipelines.pt_utils import KeyDataset
from typing import Optional, Union
import yaml

from travel import set_random_seed
from travel.constants import CACHE_FREQUENCY, RANDOM_SEED, CONFIG_PATH
from travel.data.vqg import VQGInputs, VQGOutputs, parse_vqg_outputs, save_vqg_outputs
from travel.model.mistake_detection import NLI_BATCH_SIZE, NLI_RELEVANCE_DELTA

# TODO: may need to reform prompts for recipe steps to include more information from the recipe - previous steps, ingredients, or recipe name? - at least for CaptainCook4D
def run_vqg(lm: TextGenerationPipeline, inputs: list[VQGInputs], input_ids: list[str], batch_size: int=8, save_path: Optional[str]=None, vqg_outputs: dict[str, VQGOutputs]={}, omit_failed_instances: bool=True) -> dict[str, Optional[VQGOutputs]]:
    """
    Runs VQG with a given LM text generation pipeline and list of VQG inputs.

    :param lm: TextGenerationPipeline from `transformers` library.
    :param inputs: List of VQG inputs, including procedures, prompts, etc.
    :param input_ids: Unique string identifiers for inputs. These may characterize a specific prompt or run of a prompt (e.g., at a different temperature).
    :param save_path: Optional path to save VQG outputs during and after running. Must either be a json filename or path to a directory.
    :param vqg_outputs: Partly filled dictionary of VQG outputs to start from; only pass this if starting from a partially completed run of VQG, and make sure complete/incomplete prompts are managed appropriately outside of this method.
    :return: Completed dictionary of VQGOutputs.
    """
    assert len(inputs) == len(input_ids), "run_vqg expected the same number of inputs and input IDs!"
    prompt_idx = 0
    with torch.no_grad():
        for inp, inp_id, out in tqdm(zip(inputs,
                                         input_ids,
                                         lm(KeyDataset([inp.to_dict() for inp in inputs], "prompt"), 
                                            batch_size=batch_size, 
                                            max_new_tokens=128, 
                                            return_full_text=False, 
                                            truncation="do_not_truncate")),
                                     desc=f"running VQG ({str(lm.device)})",
                                     total=len(inputs)):

            procedure_id = int(inp.procedure_id)
            step = inp.procedure_description

            # During VQG, set random seed based on procedure ID to ensure we don't use the same random seed in other parallel processes
            # TODO: this doesn't work as expected, fix
            set_random_seed(RANDOM_SEED * procedure_id)

            text = out[0]['generated_text']
            
            # Hack: sometimes output from LLaMA 2 starts with Љ and whitespace characters
            text_fixed = text.replace("Љ", "").strip() 
            
            # Parse reported target object and questions and answers
            try:
                output = parse_vqg_outputs(text_fixed, procedure_id, step)
            except:
                print("Warning: failed to parse a VQG output.")
                print(text)
                print('======')
                print(text_fixed)  

                if not omit_failed_instances:
                    vqg_outputs[inp_id] = None

                continue

            vqg_outputs[inp_id] = output

            del out

            if prompt_idx % CACHE_FREQUENCY == 0 and save_path is not None:
                print("Saving progress...")
                save_vqg_outputs(vqg_outputs, save_path)

            prompt_idx += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save progress one last time after completion
    if save_path is not None:
        save_vqg_outputs(vqg_outputs, save_path)
    return vqg_outputs

def run_vqg_semi_structured(lm: TextGenerationPipeline, inputs: list[VQGInputs], input_ids: list[str], batch_size: int=8, save_path: Optional[str]=None, vqg_outputs: dict[str, VQGOutputs]={}, omit_failed_instances: bool=True) -> dict[str, Optional[VQGOutputs]]:
    """
    Runs VQG with a given LM text generation pipeline and list of VQG inputs. Uses a semi-structured approach to ensure structure of LM outputs can be parsed later.

    :param lm: TextGenerationPipeline from `transformers` library.
    :param inputs: List of VQG inputs, including procedures, prompts, etc.
    :param input_ids: Unique string identifiers for inputs. These may characterize a specific prompt or run of a prompt (e.g., at a different temperature).
    :param save_path: Optional path to save VQG outputs during and after running. Must either be a json filename or path to a directory.
    :param vqg_outputs: Partly filled dictionary of VQG outputs to start from; only pass this if starting from a partially completed run of VQG, and make sure complete/incomplete prompts are managed appropriately outside of this method.
    :return: Completed dictionary of VQGOutputs.
    """
    def run_vqg_partial(inputs, input_ids, lm):
        partial_outputs = []
        with torch.no_grad():
            for inp, inp_id, out in tqdm(zip(inputs,
                                            input_ids,
                                            lm(KeyDataset([inp.to_dict() for inp in inputs], "prompt"), 
                                                batch_size=batch_size, 
                                                max_new_tokens=32, 
                                                return_full_text=False, 
                                                truncation="do_not_truncate")),
                                        desc=f"running VQG ({str(lm.device)})",
                                        total=len(inputs)):

                # TODO: fix random seed functionality here too

                text = out[0]['generated_text']
                
                # Hack: sometimes output from LLaMA 2 starts with Љ and whitespace characters
                text_fixed = text.replace("Љ", "").strip()
                partial_outputs.append(text_fixed)
        return partial_outputs

    assert len(inputs) == len(input_ids), "run_vqg expected the same number of inputs and input IDs!"

    # Get first question
    for i in range(len(inputs)):
        inputs[i].prompt += "1. "
    partial_outputs = run_vqg_partial(inputs, input_ids, lm)

    # Get first answer
    for i, text in enumerate(partial_outputs):
        inputs[i].prompt += text.split("?")[0] + "? (yes/no) "
    partial_outputs = run_vqg_partial(inputs, input_ids, lm)

    # Get second question
    for i, text in enumerate(partial_outputs):
        inputs[i].prompt += " " + text.split()[0] + "\n2. "
    partial_outputs = run_vqg_partial(inputs, input_ids, lm)

    # Get second answer
    for i, text in enumerate(partial_outputs):
        inputs[i].prompt += text.split("?")[0] + "? (yes/no) "
    partial_outputs = run_vqg_partial(inputs, input_ids, lm)

    for inp, inp_id, out in zip(inputs, input_ids, partial_outputs):
        final_text = inp.prompt + " " + out.split()[0]
        try:
            output = parse_vqg_outputs(final_text, inp.procedure_id, inp.procedure_description)
        except:
            print("Warning: failed to parse a VQG output.")
            print(final_text)

            if not omit_failed_instances:
                vqg_outputs[inp_id] = None

            continue

        vqg_outputs[inp_id] = output

    # Save progress one last time after completion
    save_vqg_outputs(vqg_outputs, save_path)
    return vqg_outputs

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
NLI_CORRECTION_MINIMUM_CONFIDENCE = config["vqg"]["nli_correction_minimum_confidence"]
def correct_vqg_outputs_with_nli(vqg_outputs: dict[Union[str, int], VQGOutputs], nli_model, nli_tokenizer) -> dict[Union[str, int], VQGOutputs]:
    """
    Uses a given Hugging Face AutoModelForSequenceClassification (or other callable that can be used in the same way) geared toward NLI to correct proposed answers in VQG outputs.
    """
    all_questions = [question for output in vqg_outputs.values() for question in output.questions]
    all_answers = [answer for output in vqg_outputs.values() for answer in output.answers]
    all_procedures = [output.procedure_description for output in vqg_outputs.values() for question in output.questions]
    all_premises_yes = [f"{question} Yes" for question in all_questions]
    all_premises_no = [f"{question} No" for question in all_questions]
    # TODO: we should also check whether answers to questions contradict each other... can do this by using one QA as premise and the other as hypothesis and vice versa

    new_answers = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_questions), NLI_BATCH_SIZE), desc=f"running NLI ({str(nli_model.device)})"):
            # Prepare the batch
            batch_questions = all_questions[i:i+NLI_BATCH_SIZE]
            batch_answers = all_answers[i:i+NLI_BATCH_SIZE]
            batch_procedures = all_procedures[i:i+NLI_BATCH_SIZE]
            batch_premises_yes = all_premises_yes[i:i+NLI_BATCH_SIZE]
            batch_premises_no = all_premises_no[i:i+NLI_BATCH_SIZE]
            
            hypothesis = [f'The procedure "{procedure}" has been successfully completed.' for procedure in batch_procedures]

            # Run premise through NLI model
            x = nli_tokenizer.batch_encode_plus(list(zip(batch_premises_yes, hypothesis)),
                                                    return_tensors='pt', 
                                                    padding="longest",
                                                    truncation='only_first')
            logits_yes = nli_model(**x.to(nli_model.device))[0]
            logits_yes = logits_yes.cpu()
            logits_yes = logits_yes[:,[0,2]] # Take logits for contradiction and entailment only
            probs_yes = logits_yes.softmax(dim=1)

            # Run negated premise through NLI model
            x = nli_tokenizer.batch_encode_plus(list(zip(batch_premises_no, hypothesis)), 
                                                    return_tensors='pt',
                                                    padding="longest",
                                                    truncation='only_first')
            logits_no = nli_model(**x.to(nli_model.device))[0]
            logits_no = logits_no.cpu()
            logits_no = logits_no[:,[0,2]] # Take logits for contradiction and entailment only
            probs_no = logits_no.softmax(dim=1)
            
            probs = torch.cat((probs_no[:, 1].unsqueeze(1), probs_yes[:, 1].unsqueeze(1)), dim=1)
            success_answers = torch.argmax(probs, dim=1)
            
            # TODO: may need to change this logic later or update threshold
            for old_a, success_answer, this_probs in zip(batch_answers, success_answers.tolist(), probs.tolist()):
                # Correct answer if NLI model is at least 80% sure and there's a big difference between success probability for yes and no
                success_prob = this_probs[success_answer]
                relevance = abs(this_probs[1] - this_probs[0])
                if success_prob >= NLI_CORRECTION_MINIMUM_CONFIDENCE and relevance >= NLI_RELEVANCE_DELTA:
                    new_answers.append("No" if success_answer == 0 else "Yes")
                else:
                    new_answers.append("No" if old_a == 0 else "Yes")

            # TODO: consider calculating relevance here later and use it to flag bad questions?
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    assert len(new_answers) == len(all_questions)
    parallel_idx = 0

    new_vqg_outputs = {}
    for output in vqg_outputs.values():
        output_new_answers = []
        for _ in output.questions:
            output_new_answers.append(new_answers[parallel_idx])
            parallel_idx += 1

        new_output = VQGOutputs(
                procedure_id=output.procedure_id,
                procedure_description=output.procedure_description,
                questions=output.questions,
                answers_str=output_new_answers
        )
        new_vqg_outputs[output.procedure_id] = new_output

    # Report how many answers were corrected
    n_corrected = 0
    total = 0
    for new_output, old_output in zip(new_vqg_outputs.values(), vqg_outputs.values()):
        for i in range(len(new_output.questions)):
            if new_output.answers[i] != old_output.answers[i]:
                n_corrected += 1
            total += 1

    print(f"Corrected {n_corrected}/{total} ({round(n_corrected/total, 3)}) VQG proposed answers!")

    return new_vqg_outputs

def cleanup_generated_question(question):
    """Cleanup method for generated questions in iterative VQA pipeline."""
    question = question.split("?")[0].strip() + "?"
    if "." in question:
        question = question.split(".")[1].strip()    
    return question