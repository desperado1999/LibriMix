import os
import argparse
import soundfile as sf
import pandas as pd
import glob
from tqdm import tqdm
from pathlib import Path

# Global Parameters
# Filter out files shorter than that
NUMBER_OF_SECONDS = 3
# The sample rate of the audio should be 16kHz
RATE = 16000

parser = argparse.ArgumentParser()
parser.add_argument(
    "--aishell1_dir", type=str, required=True, help="Path to aishell1 root directory"
)


def main(args):
    aishell1_dir = args.aishell1_dir
    aishell1_md_dir = os.path.join(aishell1_dir, "metadata")
    os.makedirs(aishell1_md_dir, exist_ok=True)
    create_aishell1_metadata(aishell1_dir, aishell1_md_dir)


def create_aishell1_metadata(aishell1_dir, md_dir):
    dir_to_process = check_aready_generated(aishell1_dir, md_dir)
    for ldir in dir_to_process:
        dir_metadata = create_aishell1_dataframe(aishell1_dir, ldir)

        num_samples = NUMBER_OF_SECONDS * RATE
        dir_metadata = dir_metadata[dir_metadata["length"] >= num_samples]
        dir_metadata = dir_metadata.sort_values("length")
        save_path = os.path.join(md_dir, ldir + ".csv")
        dir_metadata.to_csv(save_path, index=False)


def create_aishell1_dataframe(aishell1_dir, subdir):
    print(
        f"Creating {subdir} metadata file in "
        f"{os.path.join(aishell1_dir, 'metadata')}"
    )
    dir_path = os.path.join(aishell1_dir, subdir)
    sound_paths = glob.glob(os.path.join(dir_path, "**/*.wav"), recursive=True)

    dir_md = pd.DataFrame(columns=["speaker_ID", "subset", "length", "origin_path"])
    for sound_path in tqdm(sound_paths, total=len(sound_paths)):
        spk_id = Path(sound_path).parent.name
        subset = subdir
        temp, sample_rate = sf.read(sound_path, samplerate=None)
        assert sample_rate == RATE, "The sample rate of the audio should be 16kHz"
        length = len(temp)
        rel_path = os.path.relpath(sound_path, aishell1_dir)
        dir_md.loc[len(dir_md)] = [spk_id, subset, length, rel_path]
    return dir_md


def check_aready_generated(aishell1_dir, md_dir):
    already_generated_csv = os.listdir(md_dir)
    already_generated_csv = [f.strip(".csv") for f in already_generated_csv]
    original_aishell1_dirs = ["dev", "test", "train"]
    actual_aishell1_dirs = (set(next(os.walk(aishell1_dir))[1]) & set(original_aishell1_dirs))

        # actual_librispeech_dirs = (set(next(os.walk(librispeech_dir))[1]) &
        #                        set(original_librispeech_dirs))

    not_already_processed_directories = list(
        set(actual_aishell1_dirs) - set(already_generated_csv)
    )

    return not_already_processed_directories


if __name__ == "__main__":
    # aishell1_dir = "/data/shared/speech/AISHELL1/"
    # meta_dir = os.path.join(aishell1_dir, "metadata")

    args = parser.parse_args()
    main(args)
