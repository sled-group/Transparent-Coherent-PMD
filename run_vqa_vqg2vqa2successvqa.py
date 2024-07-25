from travel import init_travel
init_travel()

import argparse
from collections import defaultdict
import concurrent.futures
from copy import deepcopy
import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
from pprint import pprint
import spacy
import torch
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from travel.constants import RESULTS_DIR
from travel.data.captaincook4d import CaptainCook4DDataset
from travel.data.ego4d import Ego4DMistakeDetectionDataset
from travel.data.mistake_detection import MistakeDetectionExample, get_cutoff_time_by_proportion, MistakeDetectionTasks
from travel.data.vqa import VQA_PROMPT_TEMPLATES, VQAResponse, SUCCESSVQA_QUESTION_TEMPLATE, CAPTION_VQA_PROMPT_TEMPLATES, VQG2VQA2SUCCESSVQA_PROMPT_TEMPLATES, get_vqa_response_token_ids, VQAOutputs
from travel.data.vqg import VQGOutputs
from travel.model.grounding import VisualFilterTypes, ContrastiveRegionFilter, TargetObjectCounterFilter, VisualContrastiveFilter
from travel.model.mistake_detection import aggregate_mistake_probs_over_frames, DETECTION_FRAMES_PROPORTION, MISTAKE_DETECTION_STRATEGIES, compile_mistake_detection_preds, generate_det_curve
from travel.model.nli import NLI_RERUN_ON_RELEVANT_EVIDENCE
from travel.model.vqa import run_vqa_for_mistake_detection

parser = argparse.ArgumentParser()
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--eval_partitions", nargs='+', type=str, default=["val"])
parser.add_argument("--mistake_detection_strategy", type=str, default="heuristic", choices=list(MISTAKE_DETECTION_STRATEGIES.keys()))
parser.add_argument("--vqg2vqa_preds_path", type=str, help="Path to output directory for previous run of VQG2VQA. The VQA outputs from this run will be used to condition VLM before prompting it for SuccessVQA. Directory must contain a preds_<eval_partition>_heuristic.json file for every partition passed in --eval_partitions.")
parser.add_argument("--mistake_detection_strategy", type=str, default="heuristic", choices=list(MISTAKE_DETECTION_STRATEGIES.keys()))
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--visual_filter_strength", type=float, required=False, default=1.0, help="Float strength for masks used in visual filters. Depending on the visual filter type, this may be interpreted as a percentage darkness or a Gaussian blur kernel size.")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for VQA inference.")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples.")
parser.add_argument("--debug", action="store_true", help="Pass this argument to run on only a small amount of data for debugging purposes.")
parser.add_argument("--debug_n_examples", type=int, default=250, help="Configure the number of examples per class to generate for debugging purposes.")
parser.add_argument("--cache_vqa_frames", action="store_true", help="Pass this argument to cache frames in VQA outputs (e.g., to inspect visual filter resuilts). This consumes a lot of disk space for large datasets.")
parser.add_argument("--caption_first", action="store_true", help="Pass this argument to first have the VLM generate a caption of the frame before answering a question.")
args = parser.parse_args()

# Load VLM(s), processors, visual filters, etc. - if multiple GPUs available, use them
print("Setting up VLMs and visual filters...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
n_workers = 1 if torch.cuda.device_count() <= 1 else torch.cuda.device_count()
vlms = []
vlm_processors = []
visual_filters = []
nlps = []
for worker_index in range(n_workers):
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{worker_index}")
    vlm = AutoModelForVision2Seq.from_pretrained(args.vlm_name, 
                                                quantization_config=bnb_config)
    vlm.language_model.generation_config.temperature = None
    vlm.language_model.generation_config.top_p = None
    vlm.language_model.generation_config.do_sample = False
    vlms.append(vlm)
        
    vlm_processor = AutoProcessor.from_pretrained(args.vlm_name)
    vlm_processor.tokenizer.padding_side = "left"
    vlm_processors.append(vlm_processor)

    if args.visual_filter_mode is not None:
        if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Contrastive_Region:
            visual_filter = ContrastiveRegionFilter(mask_strength=args.visual_filter_strength, device=f"cuda:{worker_index}")
            nlp = spacy.load('en_core_web_lg')
        if VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Visual_Contrastive:
            visual_filter = VisualContrastiveFilter(alpha=args.visual_filter_strength, device=f"cuda:{worker_index}")
            nlp = spacy.load('en_core_web_lg')            
        elif VisualFilterTypes(args.visual_filter_mode) == VisualFilterTypes.Target_Object_Counter:
            visual_filter = TargetObjectCounterFilter(device=f"cuda:{worker_index}")
            nlp = spacy.load('en_core_web_lg')      
        else:
            raise NotImplementedError(f"Visual filter type {args.visual_filter_mode} is not compatible with SuccessVQA!")
        
        visual_filters.append(visual_filter)
        nlps.append(nlp)
    else:
        visual_filters.append(None)
        nlps.append(None)

timestamp = datetime.datetime.now()
this_results_dir = os.path.join("ego4d_debug250", args.vlm_name.split("/")[-1], f"VQG2VQA2SuccessVQA_ego4d_debug250")
this_results_dir += f"_{args.vlm_name.split('/')[-1]}"
if visual_filter is not None:
    this_results_dir += f"_spatial_norephrase1.0"
this_results_dir += f"_{timestamp.strftime('%Y%m%d%H%M%S')}"
this_results_dir = os.path.join(RESULTS_DIR, "vqa_mistake_detection", this_results_dir)
os.makedirs(this_results_dir)

vqa_prompt_template = VQA_PROMPT_TEMPLATES[type(vlm)]
successvqa_question_template = SUCCESSVQA_QUESTION_TEMPLATE
response_token_ids = get_vqa_response_token_ids(vlm_processor.tokenizer)
vqg2vqa2successvqa_prompt_template = VQG2VQA2SUCCESSVQA_PROMPT_TEMPLATES[type(vlm)]
def generate_prompts(example: MistakeDetectionExample, add_caption_placeholder: bool=False):
    if add_caption_placeholder:
        raise NotImplementedError("caption_first mode not supported yet!")
        
    questions = []
    prompts = []
    answers = []
    frames = []
    successvqa_question = successvqa_question_template.format(step=example.procedure_description)

    expected_answer = VQAResponse["Yes"]

    for frame_idx, frame in enumerate(example.frames):
        if example.procedure_id in vqg_outputs:
            assert example.example_id in vqa_answer_probs
            
            q1 = vqg_outputs[example.procedure_id].questions[0]
            a1 = VQAResponse.Yes if vqa_answer_probs[example.example_id][frame_idx][0][VQAResponse.Yes] >= 0.5 else VQAResponse.No
            
            q2 = vqg_outputs[example.procedure_id].questions[1]
            a2 = VQAResponse.Yes if vqa_answer_probs[example.example_id][frame_idx][1][VQAResponse.Yes] >= 0.5 else VQAResponse.No
            
            prompt = vqg2vqa2successvqa_prompt_template.format(question1=q1, answer1=a1.name, question2=q2, answer2=a2.name, step=example.procedure_description)
        else:
            prompt = vqa_prompt_template.format(question=successvqa_question)

        questions.append(successvqa_question)
        prompts.append(prompt)
        answers.append(expected_answer)
        frames.append(frame)

    return questions, prompts, answers, frames

for eval_partition in args.eval_partitions:
    # Load mistake detection dataset
    if MistakeDetectionTasks(args.task) == MistakeDetectionTasks.CaptainCook4D:
        eval_datasets = [CaptainCook4DDataset(data_split=eval_partition, debug_n_examples_per_class=args.debug_n_examples if args.debug else None) for _ in range(n_workers)]
    elif MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D:
        eval_datasets = [Ego4DMistakeDetectionDataset(data_split=eval_partition, 
                                                      mismatch_augmentation=True,
                                                      multi_frame=True,
                                                      debug_n_examples_per_class=args.debug_n_examples if args.debug else None) for _ in range(n_workers)]
    elif MistakeDetectionTasks(args.task) == MistakeDetectionTasks.Ego4D_Single:
        eval_datasets = [Ego4DMistakeDetectionDataset(data_split=eval_partition, 
                                                      mismatch_augmentation=True,
                                                      multi_frame=False,
                                                      debug_n_examples_per_class=args.debug_n_examples if args.debug else None) for _ in range(n_workers)]
    else:
        raise NotImplementedError(f"Haven't implemented usage of {args.task} dataset yet!")                                        


    preds_vqg2vqa = json.load(open(os.path.join(args.vqg2vqa_preds_path, f"preds_heuristic_{eval_partition}.json"), "r"))

    # Gather up example dirs by example ID
    dataset_cache_dir = Ego4DMistakeDetectionDataset.get_cache_dir("val",
                                                            mismatch_augmentation=True,
                                                            multi_frame=True,
                                                            debug_n_examples_per_class=None)
    dataset_example_dirs = json.load(open(os.path.join(dataset_cache_dir, "dataset.json"), "r"))["example_dirs"]
    dataset_example_dirs = {"/".join(d.split("/")[-3:]): d for d in dataset_example_dirs}

    vqg_outputs = {}
    vqa_answer_probs = defaultdict(list)

    # Gather up VQG2VQA preds, including VQG outputs they were based on
    for pred in tqdm(preds_vqg2vqa.values()):
            
        example_id = pred['example']['example_id']
        example = Ego4DMistakeDetectionDataset.load_example_from_file(dataset_example_dirs[example_id])
        example.cutoff_to_last_frames(DETECTION_FRAMES_PROPORTION)
        assert len(example.frames) == len(pred['vqa'])

        # Extract the vqa list - just take the questions and answers for last frame
        # Create a figure with a subplot for each entry
        num_entries = len(pred['vqa'])
        num_questions = len(pred['vqa'][0])

        for frame_idx in range(len(pred['vqa'])):
            vqa_list = pred['vqa'][frame_idx]

            this_questions = []
            this_answers = []
            this_answer_probs = []
            for i, entry in enumerate(vqa_list):
                this_questions.append(entry['question'])
                this_answers.append(VQAResponse(int(entry['expected_answer'])).name)
                this_answer_probs.append({VQAResponse(int(k)): v for k, v in entry['answer_probs'].items()})
                
            if frame_idx == 0:
                vqg_outputs[example.procedure_id] = VQGOutputs(
                    procedure_id=example.procedure_id,
                    procedure_description=example.procedure_description,
                    questions=this_questions,
                    answers_str=this_answers,
                )
                
            vqa_answer_probs[example.example_id].append(this_answer_probs)
                
    if torch.cuda.device_count() >= 2: 
        print(f"Running VQA in parallel across {torch.cuda.device_count()} GPUs...")
        with concurrent.futures.ThreadPoolExecutor() as executor:   
            partitions = list(executor.map(run_vqa_for_mistake_detection, 
                                           eval_datasets,
                                           vlms,
                                           vlm_processors,
                                           [generate_prompts] * n_workers,
                                           [1] * n_workers,
                                           [None if args.visual_filter_mode is None else VisualFilterTypes(args.visual_filter_mode)] * n_workers,
                                           visual_filters,
                                           nlps,
                                           [this_results_dir] * n_workers,
                                           [n_workers] * n_workers,
                                           list(range(n_workers)),
                                           [args.batch_size] * n_workers,
                                           [args.cache_vqa_frames] * n_workers,
                                           [args.caption_first] * n_workers)
                             )
        # Compile processed data from partitions
        vqa_outputs = []
        for this_vqa_outputs in partitions:
            vqa_outputs += this_vqa_outputs        
    else:
        print("Running VQA sequentially...")
        vqa_outputs = run_vqa_for_mistake_detection(eval_dataset=eval_datasets[0],
                                                    vlm=vlms[0],
                                                    vlm_processor=vlm_processors[0],
                                                    generate_prompts=generate_prompts,
                                                    n_prompts_per_frame=1,
                                                    visual_filter_mode=None if args.visual_filter_mode is None else VisualFilterTypes(args.visual_filter_mode),
                                                    visual_filter=visual_filters[0],
                                                    nlp=nlps[0],
                                                    cache_dir=this_results_dir,
                                                    n_workers=1,
                                                    worker_index=0,
                                                    vqa_batch_size=args.batch_size,
                                                    cache_frames=args.cache_vqa_frames,
                                                    caption_first=args.caption_first)
    
    print("Combining VQG2VQA and SuccessVQA outputs...")
    new_vqa_outputs = deepcopy(vqa_outputs)
    for example_idx in tqdm(range(len(new_vqa_outputs))):
        for frame_idx in range(len(new_vqa_outputs[example_idx])):
            # Take the probability of most likely answer for each question in VQG2VQA
            this_probs = []
            for question_probs in vqa_answer_probs[vqa_outputs[example_idx][frame_idx][0].example_id][frame_idx]:
                this_probs.append(max(question_probs.values()))

            # Average SuccessVQA probability with the probabilities of those answers included from VQG2VQA outputs
            new_yes_prob = new_vqa_outputs[example_idx][frame_idx][0].answer_probs[VQAResponse.Yes]
            for p in this_probs:
                new_yes_prob += p
            new_yes_prob /= 3.0
            answer_probs = {VQAResponse.Yes: new_yes_prob, VQAResponse.No: 1.0 - new_yes_prob}
            new_vqa_outputs[example_idx][frame_idx][0].answer_probs = answer_probs

    print("Evaluating and saving results...")        
    evaluator = MISTAKE_DETECTION_STRATEGIES[args.mistake_detection_strategy](eval_datasets[0], vqa_outputs)
    mistake_detection_preds, metrics = evaluator.evaluate_mistake_detection()
    print(f"Mistake Detection Metrics ({eval_partition}, Detection Threshold={metrics['accuracy']['best_threshold']}):")
    pprint(metrics[metrics['accuracy']['best_threshold']])

    # Compile preds per mistake detection example
    preds = compile_mistake_detection_preds(eval_datasets[0], vqa_outputs, mistake_detection_preds, image_base_path=this_results_dir)

    # Save metrics, preds, DET curve, config file (which may have some parameters that vary over time), and command-line arguments
    metrics_filename = f"metrics_{args.mistake_detection_strategy}{'rerun' if args.mistake_detection_strategy == 'nli' and NLI_RERUN_ON_RELEVANT_EVIDENCE else ''}_{eval_partition}.json"
    json.dump(metrics, open(os.path.join(this_results_dir, metrics_filename), "w"), indent=4)

    preds_filename = f"preds_{args.mistake_detection_strategy}{'rerun' if args.mistake_detection_strategy == 'nli' and NLI_RERUN_ON_RELEVANT_EVIDENCE else ''}_{eval_partition}.json"
    json.dump(preds, open(os.path.join(this_results_dir, preds_filename), "w"), indent=4)

    det_filename = f"det_{args.mistake_detection_strategy}{'rerun' if args.mistake_detection_strategy == 'nli' and NLI_RERUN_ON_RELEVANT_EVIDENCE else ''}_{eval_partition}.pdf"
    generate_det_curve(metrics['accuracy'], os.path.join(this_results_dir, det_filename))
