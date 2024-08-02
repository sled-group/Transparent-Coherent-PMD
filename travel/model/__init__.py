import numpy as np
from tqdm import tqdm

def simple_lm_prompt_beam_search(lm, tokenizer, prompts, max_new_tokens=20, batch_size=20, generation_kwargs={}):
    """
    Stripped down, generic LM prompting method. This method is only tested for constrained beam search, and might not work for other generation settings.
    """
    if lm.generation_config.num_return_sequences:
        num_seq = lm.generation_config.num_return_sequences
    elif "num_return_sequences" in generation_kwargs:
        num_seq = generation_kwargs["num_return_sequences"]
    else:
        num_seq = 1

    all_outputs = []
    all_scores = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"running generation ({str(lm.device)})"):
        # Prepare the batch
        batch_prompts = prompts[i:i+batch_size]

        inputs = tokenizer(text=batch_prompts, padding=True, return_tensors="pt")
        inputs = inputs.to(lm.device)

        outputs = lm.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, **generation_kwargs)

        scores = lm.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices).cpu().numpy()
        all_scores += [round(float(np.exp2(np.mean(s))), 6) for s in scores] # Save sequence probability
        outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        
        all_outputs += [output.replace(batch_prompts[output_idx // num_seq], "") for output_idx, output in enumerate(outputs)]
        # TODO: return likelihoods from simple_prompt_lm

    # Collate generated texts and scores from beam search
    all_outputs_collated = []
    all_scores_collated = []
    for beam_search_idx in range(len(all_outputs) // 4):
        this_outputs = []
        this_scores = []
        for i in range(num_seq):
            this_outputs.append(all_outputs[beam_search_idx * len(all_outputs) // 4 + i])
            this_scores.append(all_scores[beam_search_idx * len(all_outputs) // 4 + i])
        all_outputs_collated.append(this_outputs)
        all_scores_collated.append(this_scores)

    return all_outputs_collated, all_scores_collated

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

        outputs = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        
        all_outputs += [output.replace(batch_prompts[output_idx], "") for output_idx, output in enumerate(outputs)]
        # TODO: return likelihoods from simple_prompt_lm

    return all_outputs
