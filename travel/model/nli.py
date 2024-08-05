import torch
from tqdm import tqdm
import yaml

from travel.constants import CONFIG_PATH

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
NLI_MODEL_PATH = config["mistake_detection_strategies"]["nli_model_path"]
NLI_BATCH_SIZE = int(config["mistake_detection_strategies"]["nli_batch_size"])
NLI_RELEVANCE_DELTA = float(config["mistake_detection_strategies"]["nli_relevance_delta"]) # Minimum difference of entailment probabilities between VQA answer and negated answer to judge question and answer as relevant for mistake detection
NLI_REPLACE_PROBS = bool(config["mistake_detection_strategies"]["nli_replace_probs"])
NLI_RERUN_ON_RELEVANT_EVIDENCE = bool(config["mistake_detection_strategies"]["nli_rerun_on_relevant_evidence"])

NLI_HYPOTHESIS_TEMPLATE = 'The procedure "{procedure}" has been successfully completed.' 

def run_nli(nli_tokenizer, nli_model, premise_hypothesis_pairs):
    with torch.no_grad():
        all_logits = None
        for i in tqdm(range(0, len(premise_hypothesis_pairs), NLI_BATCH_SIZE), desc=f"running NLI ({str(nli_model.device)})"):
            batch_premise_hypothesis_pairs = premise_hypothesis_pairs[i:i+NLI_BATCH_SIZE]
            
            x = nli_tokenizer.batch_encode_plus(batch_premise_hypothesis_pairs, 
                                                return_tensors='pt',
                                                padding="longest",
                                                truncation='only_first')
            logits = nli_model(**x.to(nli_model.device))[0]
            logits = logits.cpu()
            logits = logits[:,[0,2]] # Take logits for contradiction and entailment only
            if all_logits is None:
                all_logits = logits
            else:
                all_logits = torch.cat((all_logits, logits), dim=0)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_logits.softmax(dim=1)