from travel import init_travel
init_travel()

from collections import defaultdict
import numpy as np
from pprint import pprint
import spacy
import torch
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from travel.data.vqg import VQG_DEMONSTRATIONS
from travel.model.metrics import question_coherence_metrics_nli
from travel.model.nli import NLI_MODEL_PATH

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH, quantization_config=bnb_config)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)
nlp = spacy.load("en_core_web_lg")

VLM_NAME = "llava-hf/llava-1.5-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

vlm = AutoModelForVision2Seq.from_pretrained(VLM_NAME, 
                                            quantization_config=bnb_config)
vlm_processor = AutoProcessor.from_pretrained(VLM_NAME)
vlm_processor.tokenizer.padding_side = "left"
vlm_processor.tokenizer.pad_token_id = vlm_processor.tokenizer.eos_token_id

procedures = [d.procedure_description for d in VQG_DEMONSTRATIONS for _ in d.questions]
questions = [q for d in VQG_DEMONSTRATIONS for q in d.questions]


metrics = question_coherence_metrics_nli(
    nli_tokenizer,
    nli_model,
    vlm_processor.tokenizer,
    vlm.language_model,
    procedures=procedures,
    questions=questions,
)

metrics = {k: np.mean(metrics[k]) for k in ['informativeness', 'relevance']}

pprint(metrics)