import os, json
from PIL import Image
from tqdm import tqdm
from typing import Any, Optional

from travel.data.captaincook4d.constants import VIDEO_DIR, ANNOTATIONS_DIR
from travel.data import MistakeDetectionExample, MistakeDetectionDataset, MistakeDetectionTasks
from travel.data.utils import generate_float_series
from travel.data.utils.video import get_video, extract_frames

class CaptainCook4DDataset(MistakeDetectionDataset):
    # TODO: adjust this class and superclass to handle "yielding" examples for better efficiency
    # TODO: adjust this class and superclass to handle data partitions
    def __init__(self, 
                 debug_n_examples_per_class: Optional[int]):
        """
        Method to initialize and load CaptainCook4D dataset.

        :param kwargs: Task-specific arguments for dataset compilation.
        """
        super().__init__(self, 
                         debug_n_examples_per_class=debug_n_examples_per_class)

    def load_examples(self, 
                      debug_n_examples_per_class: Optional[int]) -> list[MistakeDetectionExample]:

        # Pick a sample video from CaptainCook4D
        all_video_files = os.listdir(VIDEO_DIR)
        video_paths = [f for f in all_video_files if f.endswith('.mp4')]
        STEP_ANNOTATIONS = json.load(open(os.path.join(ANNOTATIONS_DIR, "annotation_json/complete_step_annotations.json"), "r"))
        ERROR_ANNOTATIONS = json.load(open(os.path.join(ANNOTATIONS_DIR, "annotation_json/error_annotations.json"), "r"))
        for error_annotation in ERROR_ANNOTATIONS:
            video_id = error_annotation['recording_id']
            STEP_ANNOTATIONS[video_id]["steps_errors"] = error_annotation["step_annotations"]

        success_examples = []
        error_examples = []
        all_examples = []
        for sample_video_path in tqdm(video_paths):
            sample_video_id = "_".join(sample_video_path.split('_')[:2])
            sample_video_path = os.path.join(VIDEO_DIR, sample_video_path)
            try:
                sample_video = get_video(sample_video_path)
            except:
                print(f"Warning: could not open video file: {sample_video_path}")
                continue

            # Load step annotations for it and display precondition/effect frames
            for step in STEP_ANNOTATIONS[sample_video_id]["steps_errors"]:
                try:
                    # Extract some keyframes for the action
                    step_duration = step['end_time'] - step['start_time']
                    step_id = int(step['step_id'])

                    # Some steps are skipped
                    if step_duration < 0.1:
                        continue

                    adjusted_start = step['start_time'] + min(step_duration * 0.05, 0.5) # Adjust the start time to be later by a maximum of 0.5 seconds
                    adjusted_end = step['end_time'] - min(step_duration * 0.3, 3) # Adjust the end time to be earlier by a maximum of 3 seconds
                    SAMPLE_FREQUENCY = 4.0
                    times = generate_float_series(adjusted_start, adjusted_end, SAMPLE_FREQUENCY) # ultimately, we'll want to look at every image frame in some regular interval to determine if there's a mistake
                    frames = extract_frames(sample_video, times)
                    frames = [Image.fromarray(frame) for frame in frames]

                    verb, procedure_description = step['description'].split("-")[0], "-".join(step['description'].split("-")[1:])

                    if "errors" in step and len(step["errors"]) > 0:               
                        mistake_type = step['errors'][0]["tag"]
                        mistake_description = step['errors'][0]['description']
                        # altered_procedure_description = step['modified_description'] # NOTE: can use this later if needed

                        # Start with only errors specific to a single step, not related to quantities
                        # Preparation error involves the wrong object(s)
                        # Technique error involves action being performed the wrong way
                        if mistake_type not in ["Preparation Error", "Technique Error"]:
                            continue

                        if len(step['errors']) > 1:
                            print("Warning: Some error information discarded from only using the first annotated error.")            

                        error_examples.append(
                            MistakeDetectionExample(
                                "captaincook4d",
                                sample_video_id,
                                step_id,
                                frames,
                                [time - min(times) for time in times],
                                procedure_description,
                                True,
                                mistake_type,
                                mistake_description
                            )
                        )
                        all_examples.append(error_examples[-1])
                        # pprint(error_examples[-1])
                    else:
                        success_examples.append(
                            MistakeDetectionExample(
                                MistakeDetectionTasks.CaptainCook4D,
                                sample_video_id,
                                step_id,
                                frames,
                                [time - min(times) for time in times],
                                procedure_description,
                                False
                            )
                        )        
                        all_examples.append(success_examples[-1])
                        # pprint(success_examples[-1])
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
        
        if debug_n_examples_per_class is not None:
            return error_examples[:debug_n_examples_per_class] + success_examples[:debug_n_examples_per_class]
        else:
            return all_examples