import json
import os
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

EGO4D_ANNOTATION_PATH = config["data"]["ego4d"]["annotation_path"]
EGO4D_SPLIT_PATHS = {
    "train": os.path.join(config["data"]["ego4d"]["split_path"], "fho_main_train.json"),
    "val": os.path.join(config["data"]["ego4d"]["split_path"], "fho_main_val.json"),
    "test": os.path.join(config["data"]["ego4d"]["split_path"], "fho_main_test.json")
}
critical_frame_path_template = os.path.join(config["data"]["ego4d"]["sampled_frames_path"], "fho_main_{partition}-critical-frames-subsample-8")
EGO4D_CRITICAL_FRAME_PATHS = {
    "train": critical_frame_path_template.format(partition="train"),
    "val": critical_frame_path_template.format(partition="val"),
    "test": critical_frame_path_template.format(partition="test"),
}
EGO4D_VIDEO_PATH = config["data"]["ego4d"]["video_path"]

EGO4D_MISMATCH_DIR = config["data"]["ego4d"]["mismatch_dir"]
EGO4D_MISMATCH_FHO2SRL_PATH = os.path.join(EGO4D_MISMATCH_DIR, "narration_mapping_fho2srl_df.csv")
EGO4D_MISMATCH_NARRATIONS_PATH = os.path.join(EGO4D_MISMATCH_DIR, "egoclip_narrations_exploed_groupby_no,txt.csv")
EGO4D_MISMATCH_NARRATIONS_ROWS_PATH = os.path.join(EGO4D_MISMATCH_DIR, "fho_narration_df_rows.json")
EGO4D_MISMATCH_GROUPS_PATH = os.path.join(EGO4D_MISMATCH_DIR, "egoclip_groups_groupby_no,txt.csv")
EGO4D_MISMATCH_COUNT = config["data"]["ego4d"]["mismatch_count"]