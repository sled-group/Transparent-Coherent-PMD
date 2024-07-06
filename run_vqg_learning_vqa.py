# Need this call at the beginning of every script to set random seeds and set the HF cache
from travel import init_travel
init_travel()

import argparse
import json
from memory_profiler import profile
import os
from pympler.tracker import SummaryTracker
import shutil
from tqdm import tqdm

from travel.constants import IMAGES_CHUNK_SIZE
from travel.data.utils import split_list_into_partitions
from travel.data.utils.image import resize_with_aspect, CACHED_FRAME_DIMENSION
from travel.data.vqa import VQAOutputs
from travel.data.vqg_learning import load_frameVQA_examples, save_vqg_training_examples, load_vqg_training_examples, FrameVQAMistakeDetectionExample, VQGTrainingExample
from travel.model.grounding import VisualFilterTypes
from travel.model.vqa import save_vqa_outputs
from travel.model.vqg_learning import FrameVQAMistakeDetectionScorer

parser = argparse.ArgumentParser()
parser.add_argument("--vqg_directory", type=str, required=True, help="Directory where desired frameVQA_examples.json is stored.")
parser.add_argument("--vlm_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Name or path to Hugging Face model for VLM.")
parser.add_argument("--partition", type=str, choices=["train", "val", "test"], help="List of partitions to generate data for.")
parser.add_argument("--visual_filter_mode", type=str, required=False, choices=[t.value for t in VisualFilterTypes], help="Visual attention filter mode.")
parser.add_argument("--visual_filter_strength", type=float, required=False, default=1.0, help="Float strength for masks used in visual filters.")
parser.add_argument("--batch_size", type=int, default=52, help="Batch size for VQA inference. Visual filter batch size is configured in `config.yml`.")
parser.add_argument("--resume_dir", type=str, help="Path to results directory for previous incomplete run of generating frameVQA examples. Can also be used to add another partition of data to existing reuslts directory.")
parser.add_argument("--track_memory", action="store_true", help="Pass this argument to use `pympler` to print out summaries of memory usage periodically during execution.")
parser.add_argument("--cache_vqa_frames", action="store_true", help="Pass this argument to cache frames in VQA outputs (e.g., to inspect visual filter resuilts). This consumes a lot of disk space for large datasets.")
parser.add_argument("--run_id", type=str, help="Unique ID for this run.")

args = parser.parse_args()

if "_debug" in args.vqg_directory:
    IMAGES_CHUNK_SIZE = 10

if args.track_memory:
    tracker = SummaryTracker() # Use pympler to track memory leaks
else:
    tracker = None

# Prepare output directory; save training examples for VQG in sub-folder of VQG results
if args.resume_dir is None:
    vlm_name = args.vlm_name.split('/')[-1]
    this_results_dir = os.path.join(args.vqg_directory, vlm_name, f"VQA_data_{vlm_name}")
    if args.visual_filter_mode is not None:
        this_results_dir += f"_{args.visual_filter_mode}{args.visual_filter_strength}"
    if args.run_id is not None:
        this_results_dir += f"_{args.run_id}"
else:
    this_results_dir = args.resume_dir

# Load outputs
print("Loading data...")
frameVQA_examples = load_frameVQA_examples(args.vqg_directory, args.partition, load_frames=False)
if "_debug" in args.vqg_directory:
    frameVQA_examples = frameVQA_examples[:200]
print(f"{len(frameVQA_examples)} frame VQA examples loaded from {args.partition} partition")

# If we have a lot of data, divide it into chunks to conserve memory
if len(frameVQA_examples) > IMAGES_CHUNK_SIZE:
    frameVQA_examples_split = split_list_into_partitions(frameVQA_examples, len(frameVQA_examples) // IMAGES_CHUNK_SIZE)
else:
    frameVQA_examples_split = [frameVQA_examples]

# Prepare cache directories and omit any directories where examples have already been saved
cache_dirs = [os.path.join(this_results_dir, f"VQA_scoring_cache_{args.partition}", f"VQA_cache_{args.partition}{global_chunk_idx}") for global_chunk_idx in range(len(frameVQA_examples_split))]
cache_dirs = [cd for cd in cache_dirs if not os.path.exists(os.path.join(cd, f"vqg_training_examples_{args.partition}.json"))]

# Get ready to run VQA scoring
def run_vqa_scoring_on_chunk(scorer: FrameVQAMistakeDetectionScorer,
                             frameVQA_examples_chunk: list[FrameVQAMistakeDetectionExample],
                             cache_dir: str,
                             cache_path: str) -> tuple[list[VQGTrainingExample], list[VQAOutputs]]:
    """Local method to run VQA scoring on a chunk of data."""
    
    # Load frames for this chunk
    for example in frameVQA_examples_chunk:
        example.uncache_frame()
    
    # Run VQA on this chunk
    this_vqg_training_examples, this_vqa_outputs = scorer(frameVQA_examples_chunk,
                                                            return_vqa_outputs=True,
                                                            batch_size=args.batch_size,
                                                            cache_path=cache_path,
                                                            memory_tracker=tracker)
    
    # Cache frames in VQAOutputs to conserve memory
    frames_to_close = []
    for outputs in this_vqa_outputs:
        for output in outputs:
            if type(output.frame) != str:
                output.frame = resize_with_aspect(output.frame, CACHED_FRAME_DIMENSION) # Since this is just for inspection purposes, save a smaller copy
                frames_to_close.append(output.frame)
                if args.cache_vqa_frames:
                    output.cache_frame(cache_dir)
                else:
                    # Setting frame to a string prevents frame from being saved in memory later
                    output.frame = ""
        
    # Close frames - have to do this afterward since some are shared between VQAOutputs
    for frame in frames_to_close:
        frame.close()

    # And close images in input examples
    for example in frameVQA_examples_chunk:
        example.frame.close()

    return this_vqg_training_examples, this_vqa_outputs

# Enable profiling by prepending `mprof run` to the command used to run this script, then use `mprof plot --output=mprof.pdf` to see PDF of memory usage plot
@profile
def run_vqa_scoring_on_chunks(scorer: FrameVQAMistakeDetectionScorer,
                                frameVQA_examples_chunks: list[list[FrameVQAMistakeDetectionExample]],
                                chunks_cache_dirs: int) -> tuple[list[VQGTrainingExample], list[VQAOutputs]]:
    """Local method to run VQA scoring on a list of chunks of data."""
    for frameVQA_examples_chunk, cache_dir in tqdm(zip(frameVQA_examples_chunks, chunks_cache_dirs), desc="chunks"):
        # if chunk_idx > 3:
        #     if args.track_memory:
        #         set_memory_limit(int(2.6214e+9)) # Set a limit of memory usage to catch memory spikes

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        this_vqg_training_examples, this_vqa_outputs = run_vqa_scoring_on_chunk(scorer=scorer,
                                                                                frameVQA_examples_chunk=frameVQA_examples_chunk,
                                                                                cache_dir=cache_dir,
                                                                                cache_path=os.path.join(cache_dir, "VQA_cache.pt"))
        if args.track_memory:
            print("\nMemory (after chunk)")
            tracker.print_diff()

        # Save progress in this chunk's subfolder
        save_vqa_outputs([output for sub_output in this_vqa_outputs for output in sub_output], cache_dir, args.partition)
        save_vqg_training_examples(this_vqg_training_examples, cache_dir, args.partition)

        del frameVQA_examples_chunk

# Split up work by srun processes; if SLURM_PROCID is not accessible, just run all the work here
if "SLURM_PROCID" in os.environ:
    worker_index = int(os.environ["SLURM_PROCID"])
    n_workers = int(os.environ["SLURM_NPROCS"])

else:
    worker_index = 0
    n_workers = 1

if worker_index >= len(frameVQA_examples_split):
    print(f"Warning: Not enough chunks to parallelize into process {args.worker_index}. Exiting process.")
else:
    local_frameVQA_chunks = split_list_into_partitions(frameVQA_examples_split, n_workers)[worker_index]
    local_cache_dirs = split_list_into_partitions(cache_dirs, n_workers)[worker_index]
    assert len(local_frameVQA_chunks) == len(local_cache_dirs), f"Mismatched number of chunks and cache directories for worker {worker_index}."

    scorer = FrameVQAMistakeDetectionScorer(args.vlm_name,
                                            visual_filter_type=VisualFilterTypes(args.visual_filter_mode) if args.visual_filter_mode is not None else None,
                                            visual_filter_strength=args.visual_filter_strength,
                                            vlm_device=0,
                                            visual_filter_device=0 if args.visual_filter_mode is not None else None)
    
    print(f"({worker_index}) Running VQA scoring over {len(local_frameVQA_chunks)} chunk(s): " + ", ".join([str(len(s)) for s in local_frameVQA_chunks]))
    run_vqa_scoring_on_chunks(scorer, 
                              local_frameVQA_chunks,
                              local_cache_dirs)

# All of the data has now been saved in subdirectories of this_results_dir; it can later be loaded from this_results_dir using load_vqg_training_examples method
if worker_index == 0:
    # Combine all the generated data


    shutil.copy("config.yml", os.path.join(this_results_dir, "config.yml"))
    json.dump(args.__dict__, open(os.path.join(this_results_dir, "args.json"), "w"), indent=4)

    # Delete extra cached logits (each of which take up about 250MB)
    for cache_dir in cache_dirs:
        os.remove(os.path.join(cache_dir, "VQA_cache.pt"))
