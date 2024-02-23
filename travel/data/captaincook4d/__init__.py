import os, json
import pickle
from PIL import Image
from tqdm import tqdm
from typing import Any, Optional

from travel.constants import DATA_CACHE_DIR
from travel.data.captaincook4d.constants import VIDEO_DIR, ANNOTATIONS_DIR, DATA_SPLITS
from travel.data.mistake_detection import MistakeDetectionExample, MistakeDetectionDataset, MistakeDetectionTasks
from travel.data.utils import generate_float_series
from travel.data.utils.video import get_video, extract_frames, FRAME_SAMPLING_FREQUENCY

class CaptainCook4DDataset(MistakeDetectionDataset):
    def __init__(self, 
                 data_split: str,
                 load_videos: bool=True,
                 debug_n_examples_per_class: Optional[int]=None):
        """
        Method to initialize and load CaptainCook4D dataset.

        :param kwargs: Task-specific arguments for dataset compilation.
        """
        super().__init__(data_split,
                         load_videos,
                         debug_n_examples_per_class=debug_n_examples_per_class)

    # TODO: don't load videos on init? Instead can just load annotations and frame times, then load actual frames when accessing specific items
    # TODO: or just consider parallelizing
    def load_examples(self,
                      data_split: str,
                      load_videos: bool=True,
                      debug_n_examples_per_class: Optional[int]=None) -> list[MistakeDetectionExample]:

        # TODO: When loading CaptainCook4D at least a few videos cannot be successfully loaded. Need to look into this at some point

        # Check if we already loaded data before
        cache_fname = f"captaincook4d_{data_split}_freq{FRAME_SAMPLING_FREQUENCY}" 
        if debug_n_examples_per_class is not None:
            cache_fname += f"_debug{debug_n_examples_per_class}"
        cache_fname = os.path.join(DATA_CACHE_DIR, cache_fname + ".pkl")

        if os.path.exists(cache_fname) and load_videos:
            error_examples, success_examples, all_examples = pickle.load(open(cache_fname, "rb"))
        else:
            # Sample videos from CaptainCook4D
            all_video_ids = DATA_SPLITS[data_split]
            all_video_paths = [os.path.join(VIDEO_DIR, f"{vid}_360p.mp4") for vid in all_video_ids]
            STEP_ANNOTATIONS = json.load(open(os.path.join(ANNOTATIONS_DIR, "annotation_json/complete_step_annotations.json"), "r"))
            ERROR_ANNOTATIONS = json.load(open(os.path.join(ANNOTATIONS_DIR, "annotation_json/error_annotations.json"), "r"))
            for error_annotation in ERROR_ANNOTATIONS:
                video_id = error_annotation['recording_id']
                STEP_ANNOTATIONS[video_id]["steps_errors"] = error_annotation["step_annotations"]

            success_examples = []
            error_examples = []
            all_examples = []
            for sample_video_id, sample_video_path in tqdm(zip(all_video_ids, all_video_paths), desc="loading captaincook4d videos", total=len(all_video_ids)):
                try:
                    sample_video = get_video(sample_video_path)
                except:
                    print(f"Warning: could not open video file: {sample_video_path}")
                    continue

                # Load step annotations for it and display precondition/effect frames
                for step_idx, step in enumerate(STEP_ANNOTATIONS[sample_video_id]["steps_errors"]):
                    try:
                        # Extract some keyframes for the action
                        step_duration = step['end_time'] - step['start_time']
                        step_id = int(step['step_id'])

                        # Some steps are skipped
                        if step_duration < 0.1:
                            continue

                        if load_videos:
                            adjusted_start = step['start_time'] + min(step_duration * 0.05, 0.5) # Adjust the start time to be later by a maximum of 0.5 seconds
                            adjusted_end = step['end_time'] - min(step_duration * 0.3, 3) # Adjust the end time to be earlier by a maximum of 3 seconds
                            times = generate_float_series(adjusted_start, adjusted_end, FRAME_SAMPLING_FREQUENCY) # ultimately, we'll want to look at every image frame in some regular interval to determine if there's a mistake
                            frames = extract_frames(sample_video, times)
                            frames = [Image.fromarray(frame) for frame in frames]
                        else:
                            times = []
                            frames = []

                        verb, procedure_description = step['description'].split("-")[0], "-".join(step['description'].split("-")[1:])

                        if "errors" in step and len(step["errors"]) > 0:               
                            mistake_type = step['errors'][0]["tag"] # TODO: group mistake types across evaluation datasets into consistent types? doesn't really matter
                            mistake_description = step['errors'][0]['description']
                            # altered_procedure_description = step['modified_description'] # NOTE: can use this later if needed

                            if len(step['errors']) > 1:
                                print("Warning: Some error information discarded from only using the first annotated error.")            

                            error_examples.append(
                                MistakeDetectionExample(
                                    task_name=MistakeDetectionTasks.CaptainCook4D,
                                    video_id=sample_video_id,
                                    procedure_id=step_id,
                                    example_id=f"{sample_video_id}_{step_idx}",
                                    frames=frames,
                                    frame_times=[time - min(times) for time in times],
                                    procedure_description=procedure_description,
                                    mistake=True,
                                    mistake_type=mistake_type,
                                    mistake_description=mistake_description
                                )
                            )
                            all_examples.append(error_examples[-1])
                            # pprint(error_examples[-1])
                        else:
                            success_examples.append(
                                MistakeDetectionExample(
                                    task_name=MistakeDetectionTasks.CaptainCook4D,
                                    video_id=sample_video_id,
                                    procedure_id=step_id,
                                    example_id=f"{sample_video_id}_{step_idx}",
                                    frames=frames,
                                    frame_times=[time - min(times) for time in times],
                                    procedure_description=procedure_description,
                                    mistake=False
                                )
                            )        
                            all_examples.append(success_examples[-1])
                    except:
                        print(f"Warning: Video {sample_video_id} step {step_id} could not be processed!")
                        continue

                if debug_n_examples_per_class is not None:
                    if len(error_examples) >= debug_n_examples_per_class and len(success_examples) >= debug_n_examples_per_class:
                        print("Collected enough positive and negative examples!")
                        break
                    else:
                        print("Error examples:", len(error_examples))
                        print("Success examples:", len(success_examples))

                sample_video.release()
        
            # If we processed videos, then cache
            if load_videos:
                pickle.dump((error_examples, success_examples, all_examples), open(cache_fname, "wb"))
            
        if debug_n_examples_per_class is not None:
            return error_examples[:debug_n_examples_per_class] + success_examples[:debug_n_examples_per_class]
        else:
            return all_examples