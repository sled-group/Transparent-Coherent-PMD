from collections import defaultdict
import os, json
from PIL import Image
from pprint import pprint
import spacy
from tqdm import tqdm
from typing import Optional

from travel.constants import DATA_CACHE_DIR
from travel.data.captaincook4d.constants import VIDEO_DIR, ANNOTATIONS_DIR, DATA_SPLITS
from travel.data.mistake_detection import MistakeDetectionExample, MistakeDetectionDataset, MistakeDetectionTasks
from travel.data.utils import generate_float_series, get_subdirectories
from travel.data.utils.image import variance_of_laplacian
from travel.data.utils.video import get_video, extract_frames, FRAME_SAMPLING_FREQUENCY, FRAME_KEEP_FREQUENCY
from travel.model.grounding import TargetObjectCounterFilter


class CaptainCook4DDataset(MistakeDetectionDataset):
    def __init__(self, 
                 data_split: str,
                 debug_n_examples_per_class: Optional[int]=None):
        """
        Method to initialize and load CaptainCook4D dataset.

        :param kwargs: Task-specific arguments for dataset compilation.
        """
        super().__init__(data_split,
                         debug_n_examples_per_class=debug_n_examples_per_class)

    def get_cache_dir(self,
                      data_split: str,
                      load_videos: bool=True,
                      debug_n_examples_per_class: Optional[int]=None) -> str:
        # Check if we already loaded data before
        cache_fname = f"captaincook4d_{data_split}_freq{FRAME_SAMPLING_FREQUENCY}-{FRAME_KEEP_FREQUENCY}" 
        if debug_n_examples_per_class is not None:
            cache_fname += f"_debug{debug_n_examples_per_class}"
        if not load_videos:
            cache_fname += "_novideos"
        cache_fname = os.path.join(DATA_CACHE_DIR, cache_fname)
        return cache_fname

    def generate_examples(self,
                          data_split: str,
                          load_videos: bool=True,
                          debug_n_examples_per_class: Optional[int]=None) -> list[MistakeDetectionExample]:

        # TODO: When loading CaptainCook4D at least a few videos cannot be successfully loaded. Need to look into this at some point

        already_processed_videos = get_subdirectories(self.cache_dir)
        print("Already processed videos:")
        pprint(already_processed_videos)

        # Sample videos from CaptainCook4D
        all_video_ids = DATA_SPLITS[data_split]
        all_video_paths = [os.path.join(VIDEO_DIR, f"{vid}_360p.mp4") for vid in all_video_ids]
        STEP_ANNOTATIONS = json.load(open(os.path.join(ANNOTATIONS_DIR, "annotation_json/complete_step_annotations.json"), "r"))
        ERROR_ANNOTATIONS = json.load(open(os.path.join(ANNOTATIONS_DIR, "annotation_json/error_annotations.json"), "r"))
        for error_annotation in ERROR_ANNOTATIONS:
            video_id = error_annotation['recording_id']
            STEP_ANNOTATIONS[video_id]["steps_errors"] = error_annotation["step_annotations"]

        # Load OWLv2 to check for target objects in recipe steps
        print("Setting up target object counter...")
        nlp = spacy.load('en_core_web_sm')
        object_counter = TargetObjectCounterFilter()

        success_examples = []
        error_examples = []
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
                    example_id = f"{sample_video_id}_{step_idx}"

                    # Check if we already processed this video
                    if example_id in already_processed_videos:
                        print(f"Skipping example {example_id} because we already processed it.")
                        continue

                    # Some steps are skipped
                    if step_duration < 0.1:
                        continue

                    procedure_description = "-".join(step['description'].split("-")[1:])

                    if load_videos:
                        adjusted_start = step['start_time'] + min(step_duration * 0.05, 0.5) # Adjust the start time to be later by a maximum of 0.5 seconds
                        adjusted_end = step['end_time'] - min(step_duration * 0.3, 3) # Adjust the end time to be earlier by a maximum of 3 seconds
                        times = generate_float_series(adjusted_start, adjusted_end, 1 / FRAME_SAMPLING_FREQUENCY) # ultimately, we'll want to look at every image frame in some regular interval to determine if there's a mistake
                        frames = extract_frames(sample_video, times)
                        frames = [Image.fromarray(frame) for frame in frames]

                        # While we initially sampled FRAME_SAMPLING_FREQUENCY frames / second, we'll only keep FRAME_KEEP_FREQUENCY frames per second - for each interval, pick a frame that is not blurry and has the maximum number of target objects
                        counts = object_counter(nlp, frames, [procedure_description] * len(frames))
                        frame_info_by_interval = defaultdict(list)
                        assert len(times) == len(frames) == len(counts), "Frames, times, and object counts must be the same."
                        for frame, time, count in zip(frames, times, counts):
                            frame_info_by_interval[int(time // (1 / FRAME_KEEP_FREQUENCY))].append((frame, time, count))
                        
                        new_frames = []
                        new_times = []
                        for interval in frame_info_by_interval:
                            max_count = max([count for _, _, count in frame_info_by_interval[interval]])
                            if max_count == 0:
                                # Skip this second of the video if there are no target objects in view
                                continue
                            frame_info_with_max_count = [info for info in frame_info_by_interval[interval] if info[2] == max_count]
                            if len(frame_info_with_max_count) > 1:
                                # If multiple frames have maximum number of target objects, take the least blurry one (max variance of laplacian)
                                frame_info_with_max_count = max(frame_info_with_max_count, key = lambda x: variance_of_laplacian(x[0]))
                            else:
                                frame_info_with_max_count = frame_info_with_max_count[0]
                            
                            new_frames.append(frame_info_with_max_count[0])
                            new_times.append(frame_info_with_max_count[1])

                        frames = new_frames
                        times = new_times
                    else:
                        times = []
                        frames = []

                    if "errors" in step and len(step["errors"]) > 0:               

                        # Filter out error types that aren't perceivable from individual images
                        mistake_types_descriptions = [m for m in step['errors'] if m['tag'] not in ["Order Error", "Timing Error", "Temperature Error"]]
                        if len(mistake_types_descriptions) > 0:
                            mistake_type, mistake_description = mistake_types_descriptions[0]['tag'], mistake_types_descriptions[0]['description']
                        else:
                            # If we didn't find any good mistake types here, omit this example
                            continue

                        # altered_procedure_description = step['modified_description'] # NOTE: can use this later if needed

                        if len(step['errors']) > 1:
                            print("Warning: Some error information discarded from only using the first annotated error.")            

                        example = MistakeDetectionExample(
                            task_name=MistakeDetectionTasks.CaptainCook4D,
                            video_id=sample_video_id,
                            procedure_id=step_id,
                            example_id=example_id,
                            frames=frames,
                            frame_times=[time - min(times) for time in times],
                            procedure_description=procedure_description,
                            mistake=True,
                            mistake_type=mistake_type,
                            mistake_description=mistake_description
                        )
                        self.save_example_to_file(example)

                        if debug_n_examples_per_class is not None:
                            error_examples.append(example)
                    else:
                        example = MistakeDetectionExample(
                            task_name=MistakeDetectionTasks.CaptainCook4D,
                            video_id=sample_video_id,
                            procedure_id=step_id,
                            example_id=example_id,
                            frames=frames,
                            frame_times=[time - min(times) for time in times],
                            procedure_description=procedure_description,
                            mistake=False
                        )
                        self.save_example_to_file(example)
                        if debug_n_examples_per_class is not None:
                            success_examples.append(example)
                    self.save_dataset_metadata()

                except Exception as e:
                    print(f"Warning: Video {sample_video_id} step {step_id} could not be processed!")
                    print(e)
                    continue

            if debug_n_examples_per_class is not None:
                if len(error_examples) >= debug_n_examples_per_class and len(success_examples) >= debug_n_examples_per_class:
                    print("Collected enough positive and negative examples!")
                    break
                else:
                    print("Error examples:", len(error_examples))
                    print("Success examples:", len(success_examples))

            sample_video.release()
      
        self.save_dataset_metadata()