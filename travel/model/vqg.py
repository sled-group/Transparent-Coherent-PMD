import torch
from tqdm import tqdm
from transformers import TextGenerationPipeline
from transformers.pipelines.pt_utils import KeyDataset
from typing import Optional

from travel.constants import CACHE_FREQUENCY
from travel.data.vqg import VQGInputs, VQGOutputs, parse_vqg_outputs, save_vqg_outputs

# TODO: include Ruixuan's improvements here... some of which are below
# TODO: may need to reform prompts for recipe steps to include more information from the recipe - previous steps, ingredients, or recipe name?
# TODO: does there need to be a single target object for VQG?
# TODO: increase number of questions to 3? or use a variable number
def run_vqg(lm: TextGenerationPipeline, inputs: list[VQGInputs], input_ids: list[str], batch_size: int=8, save_path: Optional[str]=None, vqg_outputs: dict[str, VQGOutputs]={}) -> dict[str, VQGOutputs]:
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
        # TODO: implement data parallelism over multiple GPUs?
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

            text = out[0]['generated_text']
            
            # Hack: sometimes output from LLaMA 2 starts with Љ and whitespace characters, and sometimes Љ replaces the first "T" in "Target object:"
            text_fixed = text.replace("Љ", "").strip() 
            if not text_fixed.startswith("Target object:") and ":" in text_fixed:
                text_fixed = "Target object: " + ":".join(text_fixed.split(":")[1:]).strip()
            
            # Parse reported target object and questions and answers
            try:
                try:
                    output = parse_vqg_outputs(text_fixed, procedure_id, step)
                except:
                    print("Warning: failed to parse a VQG output.")
                    continue
            except:
                print("Error parsing VQG outputs:")
                print(text)
                print('======')
                print(text_fixed)
                raise

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

