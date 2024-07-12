import torch
from tqdm import tqdm
from transformers import TextGenerationPipeline
from transformers.pipelines.pt_utils import KeyDataset
from typing import Optional

from travel import set_random_seed
from travel.constants import CACHE_FREQUENCY, RANDOM_SEED
from travel.data.vqg import VQGInputs, VQGOutputs, parse_vqg_outputs, save_vqg_outputs

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

