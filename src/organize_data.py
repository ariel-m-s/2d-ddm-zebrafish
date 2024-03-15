"""
This script organizes the data from the original structure obtained from
the fish-tracking software to a new structure that is easier to work with.

The original structure is as follows:
```
data
├── computer1
│   ├── fish1
│   │   ├── raw_data
│   │   │   ├── fish_trial1.dat
│   │   │   ├── fish_trial2.dat
│   │   │   ├── ...
│   │   ├── experiment_information.txt
│   ├── fish2
│   │   ├── raw_data
│   │   │   ├── fish_trial1.dat
│   │   │   ├── fish_trial2.dat
│   │   │   ├── ...
│   │   ├── experiment_information.txt
│   ├── ...
├── computer2
│   ├── fish1
│   │   ├── raw_data
│   │   │   ├── fish_trial1.dat
│   │   │   ├── fish_trial2.dat
│   │   │   ├── ...
│   │   ├── experiment_information.txt
│   ├── fish2
│   │   ├── raw_data
│   │   │   ├── fish_trial1.dat
│   │   │   ├── fish_trial2.dat
│   │   │   ├── ...
│   │   ├── experiment_information.txt
│   ├── ...
├── ...
```

The new structure is as follows:
```
data
├── trial1.dat
├── trial2.dat
├── ...
├── trials.csv
```

The `trials.csv` file contains the metadata for each trial in the following format:
```
trial_id,setup_id,fish_num,trial_num,dish_index,datetime,fish_age,fish_genotype
1,1,1,1,1,2021-01-01 12:00:00,1,WT
2,1,1,2,1,2021-01-01 12:30:00,1,WT
...
```

The script takes two arguments:
1. The path to the original data directory.
2. The path to the new data directory.

The script reads the original data directory, extracts the trial data, and saves it in the new data directory.
It also creates a `trials.csv` file containing the metadata for each trial.

To run the script, use the following command:
    python organize_data.py <original_data_dir> <new_data_dir>

Example:
    python src/organize_data.py data/raw data/organized
"""

import datetime as dt
import os
import shutil
import sys


def _get_dirs(path):
    """
    Get a list of directories in the given path, excluding files.

    Args:
        path (str): The path to the base directory.

    Returns:
        list: A list of directories in the given path.
    """
    # Get all directories in the path. This includes files as well.
    all_dirs = os.listdir(path)

    # Filter out files from the list of directories. Return only directories.
    return [dir for dir in all_dirs if (not os.path.isfile(os.path.join(path, dir)))]


if __name__ == "__main__":
    # Receive the source and destination directories as command-line arguments. The source directory
    # contains the original data, and the destination directory is where the organized data will be saved.
    # The source directory is the first argument, and the destination directory is the second argument.
    src = sys.argv[1]
    dst = sys.argv[2]

    # Initialize the trial count and metadata dictionary (for all trials).
    trial_count = 0
    metadata = {}

    # Get the list of computers used in the experiment.
    computers = _get_dirs(src)

    # Iterate over each computer directory.
    for computer in computers:
        # Get the list of fish directories in the computer directory.
        computer_path = os.path.join(src, computer)
        fishes = _get_dirs(computer_path)

        # Iterate over each fish directory.
        for fish in fishes:
            # Get the list of trials for the fish.
            fish_path = os.path.join(computer_path, fish)
            trials_path = os.path.join(fish_path, "raw_data")
            trials = os.listdir(trials_path)

            # Iterate over each trial for the fish.
            for trial in trials:
                trial_count += 1
                trial_id = trial_count

                # Copy the trial data to the destination directory.
                trial_path__src = os.path.join(trials_path, trial)
                trial_path__dst = os.path.join(dst, f"{trial_id}.dat")
                shutil.copy(trial_path__src, trial_path__dst)

                # Initialize the metadata for the trial.
                trial_metadata = {}

                # Determine the trial number.
                trial_num = int(trial.split(".")[0].replace("trial", ""))
                trial_metadata["trial_num"] = trial_num

                # Read the experiment information file to get additional metadata.
                with open(os.path.join(fish_path, "experiment_information.txt"), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Setup ID"):
                            setup_data = line.split(": ")[1].strip().split(" ")
                            setup_id = int(setup_data[0])
                            trial_metadata["setup_id"] = setup_id

                        elif line.startswith("Date and time"):
                            datetime_data = line.split(": ")[1].strip().split(" ")
                            dt_object = dt.datetime.strptime(datetime_data[0], "%Y-%m-%d_%H-%M-%S")
                            trial_metadata["datetime"] = str(dt_object)

                        elif line.startswith("fish_num"):
                            fish_num = int(line.split(": ")[1].strip())
                            trial_metadata["fish_num"] = fish_num

                        elif line.startswith("Fish Index"):
                            dish_data = line.split(": ")[1].strip().split(" ")
                            dish_index = int(dish_data[0])
                            trial_metadata["dish_index"] = dish_index

                        elif line.startswith("fish_age"):
                            fish_age = int(line.split(": ")[1].strip())
                            trial_metadata["fish_age"] = fish_age

                        elif line.startswith("fish_genotype"):
                            fish_genotype = line.split(": ")[1].strip()
                            trial_metadata["fish_genotype"] = fish_genotype

                # Save the metadata for the trial.
                metadata[trial_id] = trial_metadata

    # Save the metadata to the 'trials.csv' file in the destination directory. The metadata is sorted by
    # setup ID, fish number, and trial number. The 'trials.csv' file contains the metadata for each trial
    # in the following format: trial_id, setup_id, fish_num, trial_num, dish_index, datetime, fish_age,
    # fish_genotype. The 'trials.csv' file is saved in the destination directory.

    metadata = dict(sorted(metadata.items(), key=lambda x: (x[1]["setup_id"], x[1]["fish_num"], x[1]["trial_num"])))

    with open(os.path.join(dst, "trials.csv"), "w") as f:
        header = ",".join(
            ["trial_id", "setup_id", "fish_num", "trial_num", "dish_index", "datetime", "fish_age", "fish_genotype"]
        )
        f.write(f"{header}\n")

        for trial_id, trial_metadata in metadata.items():
            setup_id = trial_metadata["setup_id"]
            fish_num = trial_metadata["fish_num"]
            trial_num = trial_metadata["trial_num"]
            dish_index = trial_metadata["dish_index"]
            datetime = trial_metadata["datetime"]
            fish_age = trial_metadata["fish_age"]
            fish_genotype = trial_metadata["fish_genotype"]

            f.write(
                f"{trial_id},{setup_id},{fish_num},{trial_num},{dish_index},{datetime},{fish_age},{fish_genotype}\n"
            )
