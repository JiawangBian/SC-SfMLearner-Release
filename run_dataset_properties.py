import traceback
import sys

import pandas as pd

sys.path.insert(1, '/home/arturo/workspace/pycharm_projects/data_loader_ml/DataLoaderML')
import os
import shutil
import time
import csv
import torch
import toml
import argparse
import numpy as np
from decord import cpu, gpu
from pathlib import Path
from data_loader_ml.tools.utils import load_test_drive_ids_from_txt_file
from data_loader_ml.tools.utils import get_videos_in_dataset
from data_loader_ml.tools.utils import get_telemetry_ffname
from dataset_properties.dataset_properties_utils import get_video_properties
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='Creates a CSV file summarizing the Yaak dataset properties.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-c", "--config",
    default="configs/dataset_properties/yaak_dataset_properties.toml",
    type=Path,
    help="TOML configuration file to load the properties of the Yaak dataset.",
)


def main():

    ####################################################################################################################
    #
    # I. Initialization.
    #
    ####################################################################################################################

    # ------------------------------------------------------------------------------------------------------------------
    # Read arguments from command line.
    # ------------------------------------------------------------------------------------------------------------------

    args = parser.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Read the contents of the TOML file.
    # ------------------------------------------------------------------------------------------------------------------

    if not args.config.is_file():
        print(f"Error reading {args.config}, No such file")
        return

    with args.config.open("r") as pfile:
        cfgs = toml.load(pfile)

    # Configuration data (dataset).
    config_dataset = cfgs["dataset"]

    # Configuration data (output).
    config_output = cfgs["output"]

    # ------------------------------------------------------------------------------------------------------------------
    # Load video files, telemetry files, and drive IDs into lists.
    # ------------------------------------------------------------------------------------------------------------------

    # If drive_ids is specified as a text file, load the data (into a list) from that file.
    drive_ids = []
    if isinstance(config_dataset["drive_ids"], str) and config_dataset["drive_ids"].endswith('.txt'):
        drive_ids = load_test_drive_ids_from_txt_file(config_dataset["drive_ids"], verbose=True)

    # Otherwise, load the data from the list of test drive ids.
    elif isinstance(config_dataset["drive_ids"], list):
        drive_ids = config_dataset["drive_ids"]

    # Get video file list and drive ids list.
    video_files_list, drive_ids_list = get_videos_in_dataset(
        dataset_path=config_dataset["rootpath"],
        test_drive_id=drive_ids,
        camera_view=config_dataset["camera_view"],
        video_extension=config_dataset["extension"],
        video_filename_suffix=config_dataset["suffix"],
        verbose=False,
    )

    # Get telemetry file list.
    telemetry_files_list = [
        Path(
            Path(video_path).parent
        ).joinpath(
            config_dataset["telemetry_filename"]
        ).as_posix() for video_path in video_files_list
    ]

    # ------------------------------------------------------------------------------------------------------------------
    # Show the contents of the lists.
    # ------------------------------------------------------------------------------------------------------------------

    N = len(video_files_list)
    for index, (video_file, telemetry_file, drive_id) in \
            enumerate(zip(video_files_list, telemetry_files_list, drive_ids_list)):

        print(f"[ {index + 1} of {N} ]")
        print(f"\t[ Drive ID ] {drive_id}")
        print(f"\t[ Video file ] {video_file}")
        print(f"\t[ Telemetry file ] {telemetry_file}")
        print(" ")

    # ------------------------------------------------------------------------------------------------------------------
    # Path where data will be stored.
    # ------------------------------------------------------------------------------------------------------------------

    save_path = config_output["dataset_properties_path"]

    # If the path does not exist, it is created.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"[ Output path ] {save_path}")
        print(" ")

    ####################################################################################################################
    #
    # II. Store dataset properties into a CSV file.
    #
    ####################################################################################################################

    output_dict = {
        "drive_id": [],
        "camera_view": [],
        "video.fps": [],
        "video.total_frames": [],
        "video.frame_shape": [],
        "video.frame_dtype": [],
        "video.perc_dynamic_content": [],
        "location": [],
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Load data from videos (MP4) and metadata (JSON).
    # ------------------------------------------------------------------------------------------------------------------

    print("[ Creating CSV file of dataset properties ]")

    N = len(video_files_list)

    pbar = tqdm(range(N))

    for index in pbar:

        # Drive IDs.
        drive_id = drive_ids_list[index]

        # Progress bar.
        pbar.set_description(f"[ Processing dataset ] Drive Id = {drive_id}")

        # Video data.
        video_file = video_files_list[index]

        # Telemetry data.
        telemetry_file = telemetry_files_list[index]

        # Compute perc_dynamic_content.
        # Get location from metadata.

        # Get video properties.
        video_props_dict = get_video_properties(video_file, ctx=cpu(0), verbose=False)

        output_dict["drive_id"].append(drive_id)
        output_dict["camera_view"].append(config_dataset["camera_view"][0])
        output_dict["video.fps"].append(video_props_dict["fps"])
        output_dict["video.total_frames"].append(video_props_dict["total_frames"])
        output_dict["video.frame_shape"].append(video_props_dict["frame_shape"])
        output_dict["video.frame_dtype"].append(video_props_dict["frame_dtype"])
        output_dict["video.perc_dynamic_content"].append(-1)
        output_dict["location"].append("Berlin")

    # ------------------------------------------------------------------------------------------------------------------
    # Storing the data on disk.
    # ------------------------------------------------------------------------------------------------------------------

    # Output CSV filename.
    output_csv_filename = config_output["dataset_properties_filename"]

    # Output CSV path/filename.
    output_csv_ffname = f"{save_path}/{output_csv_filename}"

    # Pandas dataframe.
    output_df = pd.DataFrame(output_dict)

    # Storing the dataframe contents in a CSV file.
    output_df.to_csv(path_or_buf=output_csv_ffname, index=False)

    print(f"[ Dataset Properties | CSV File ] Saved in: {output_csv_ffname}")
    print(" ")


if __name__ == '__main__':
    main()
