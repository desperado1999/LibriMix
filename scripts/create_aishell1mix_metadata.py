import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

"""
    Issues:
        [1]: We have not considered the environmental noise (Wham Noise) for this moment 
"""

# Global parameters
# eps secures log and division
EPS = 1e-10
# max amplitude in sources and mixtures
MAX_AMP = 0.9
# In AISHELL1 all the sources are at 16K Hz
RATE = 16000
# We will randomize loudness between this range
MIN_LOUDNESS = -33
MAX_LOUDNESS = -25

# A random seed is used for reproducibility
random.seed(72)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--aishell1_dir', type=str, required=True,
                    help='Path to aishell1 root directory')
parser.add_argument('--aishell1_md_dir', type=str, required=True,
                    help='Path to aishell1 metadata directory')
parser.add_argument('--metadata_outdir', type=str, default=None,
                    help='Where aishell1mix metadata files will be stored.')
parser.add_argument('--n_src', type=int, required=True,
                    help='Number of sources desired to create the mixture')


def main(args):
    aishell1_dir = args.aishell1_dir
    aishell1_md_dir = args.aishell1_md_dir
    n_src = args.n_src
    # Create Librimix metadata directory
    md_dir = args.metadata_outdir
    if md_dir is None:
        root = os.path.dirname(aishell1_dir)
        md_dir = os.path.join(root, f'LibriMix/metadata')
    os.makedirs(md_dir, exist_ok=True)
    create_aishell1mix_metadata(aishell1_dir, aishell1_md_dir, md_dir, n_src)
    

def create_aishell1mix_metadata(aishell1_dir, aishell1_md_dir, md_dir, n_src):
    """ Generate aishell1mix metadata according aishell1 metadata """
    dataset = f'aishell1_{n_src}mix'
    aishell1_md_files = os.listdir(aishell1_md_dir)
    
    # If you wish to ignore some metadata files add their name here
    # Example : to_be_ignored = ['dev.csv']
    to_be_ignored = []
    aishell1_md_files = check_already_generated(md_dir, dataset, to_be_ignored, aishell1_md_files)

    for aishell1_md_file in aishell1_md_files:
        
        if not aishell1_md_file.endswith('.csv'):
            print(f"{aishell1_md_file} is not a csv file, continue.")
            continue
        aishell1_md = pd.read_csv(os.path.join(aishell1_md_dir, aishell1_md_file), engine='python')
        
        save_path =  os.path.join(md_dir, '_'.join([dataset, aishell1_md_file]))
        info_name = '_'.join([dataset, aishell1_md_file.strip('.csv'), 'info']) + '.csv'
        info_save_path = os.path.join(md_dir, info_name)
        
        
        print(f"Creating {os.path.basename(save_path)} file in {md_dir}")
        # Create dataframe
        mixtures_md, mixtures_info = create_librimix_df(
            aishell1_md, aishell1_dir, n_src)
        
        # Round number of files
        mixtures_md = mixtures_md[:len(mixtures_md) // 100 * 100]
        mixtures_info = mixtures_info[:len(mixtures_info) // 100 * 100]

        # Save csv files
        mixtures_md.to_csv(save_path, index=False)
        mixtures_info.to_csv(info_save_path, index=False)
    
def create_librimix_df(aishell1_md_file, aishell1_dir, n_src):
    """ Generate librimix dataframe from a LibriSpeech and wha md file"""

    # Create a dataframe that will be used to generate sources and mixtures
    mixtures_md = pd.DataFrame(columns=['mixture_ID'])
    # Create a dataframe with additional infos.
    mixtures_info = pd.DataFrame(columns=['mixture_ID'])
    # Add columns (depends on the number of sources)
    for i in range(n_src):
        mixtures_md[f"source_{i + 1}_path"] = {}
        mixtures_md[f"source_{i + 1}_gain"] = {}
        mixtures_info[f"speaker_{i + 1}_ID"] = {}
    # Generate pairs of sources to mix
    pairs = set_pairs(aishell1_md_file, n_src)
    
    clip_counter = 0
    # For each combination create a new line in the dataframe
    for pair in tqdm(pairs, total=len(pairs)):
        # return infos about the sources, generate sources
        sources_info, sources_list_max = read_sources(
            aishell1_md_file, pair, n_src, aishell1_dir)
        # compute initial loudness, randomize loudness and normalize sources
        loudness, _, sources_list_norm = set_loudness(sources_list_max)
        # Do the mixture
        mixture_max = mix(sources_list_norm)
        # Check the mixture for clipping and renormalize if necessary
        renormalize_loudness, did_clip = check_for_cliping(mixture_max,
                                                           sources_list_norm)
        clip_counter += int(did_clip)
        # Compute gain
        gain_list = compute_gain(loudness, renormalize_loudness)

        # Add information to the dataframe
        row_mixture, row_info = get_row(sources_info, gain_list, n_src)
        mixtures_md.loc[len(mixtures_md)] = row_mixture
        mixtures_info.loc[len(mixtures_info)] = row_info
    print(f"Among {len(mixtures_md)} mixtures, {clip_counter} clipped.")
    return mixtures_md, mixtures_info

def get_row(sources_info, gain_list, n_src):
    """ Get new row for each mixture/info dataframe """
    row_mixture = [sources_info['mixtures_id']]
    row_info = [sources_info['mixtures_id']]
    for i in range(n_src):
        row_mixture.append(sources_info['path_list'][i])
        row_mixture.append(gain_list[i])
        row_info.append(sources_info['speaker_id_list'][i])
    return row_mixture, row_info

def compute_gain(loudness, renormalize_loudness):
    """ Compute the gain between the original and target loudness"""
    gain = []
    for i in range(len(loudness)):
        delta_loudness = renormalize_loudness[i] - loudness[i]
        gain.append(np.power(10.0, delta_loudness / 20.0))
    return gain

def check_for_cliping(mixture_max, sources_list_norm):
    """Check the mixture (mode max) for clipping and re normalize if needed."""
    # Initialize renormalized sources and loudness
    renormalize_loudness = []
    clip = False
    # Recreate the meter
    meter = pyln.Meter(RATE)
    # Check for clipping in mixtures
    if np.max(np.abs(mixture_max)) > MAX_AMP:
        clip = True
        weight = MAX_AMP / np.max(np.abs(mixture_max))
    else:
        weight = 1
    # Renormalize
    for i in range(len(sources_list_norm)):
        new_loudness = meter.integrated_loudness(sources_list_norm[i] * weight)
        renormalize_loudness.append(new_loudness)
    return renormalize_loudness, clip

def mix(sources_list_norm):
    """ Do the mixture for min mode and max mode """
    # Initialize mixture
    mixture_max = np.zeros_like(sources_list_norm[0])
    for i in range(len(sources_list_norm)):
        mixture_max += sources_list_norm[i]
    return mixture_max

def set_loudness(sources_list):
    """ Compute original loudness and normalise them randomly """
    # Initialize loudness
    loudness_list = []
    # In aishell1 all sources are at 16KHz hence the meter
    meter = pyln.Meter(RATE)
    # Randomize sources loudness
    target_loudness_list = []
    sources_list_norm = []

    # Normalize loudness
    for i in range(len(sources_list)):
        # Compute initial loudness
        loudness_list.append(meter.integrated_loudness(sources_list[i]))
        # Pick a random loudness
        target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # Normalize source to target loudness

        with warnings.catch_warnings():
            # We don't want to pollute stdout, but we don't want to ignore
            # other warnings.
            warnings.simplefilter("ignore")
            src = pyln.normalize.loudness(sources_list[i], loudness_list[i],
                                          target_loudness)
        # If source clips, renormalize
        if np.max(np.abs(src)) >= 1:
            src = sources_list[i] * MAX_AMP / np.max(np.abs(sources_list[i]))
            target_loudness = meter.integrated_loudness(src)
        # Save scaled source and loudness.
        sources_list_norm.append(src)
        target_loudness_list.append(target_loudness)
    return loudness_list, target_loudness_list, sources_list_norm

def read_sources(metadata_file, pair, n_src, aishell1_dir):
    # Read lines corresponding to pair
    sources = [metadata_file.iloc[pair[i]] for i in range(n_src)]
    # Get sources info
    speaker_id_list = [source['speaker_ID'] for source in sources]
    length_list = [source['length'] for source in sources]
    path_list = [source['origin_path'] for source in sources]
    
    # Get mixtures id
    id_l = [os.path.split(source['origin_path'])[1].strip('.wav')
            for source in sources]
    mixtures_id = "_".join(id_l)

    # Get the longest and shortest source len
    max_length = max(length_list)
    sources_list = []

    # Read the source and compute some info
    for i in range(n_src):
        source = metadata_file.iloc[pair[i]]
        absolute_path = os.path.join(aishell1_dir,
                                     source['origin_path'])
        s, _ = sf.read(absolute_path, dtype='float32')
        sources_list.append(
            np.pad(s, (0, max_length - len(s)), mode='constant'))

    sources_info = {'mixtures_id': mixtures_id,
                    'speaker_id_list': speaker_id_list, 'path_list': path_list}
    return sources_info, sources_list

def set_pairs(aishell1_md_file, n_src):
    """ set pairs of sources to make the mixture """
    # Initialize list for pairs sources
    utt_pairs = []
    # In train sets utterance are only used once
    if 'train' in aishell1_md_file.iloc[0]['subset']:
        utt_pairs = set_utt_pairs(aishell1_md_file, utt_pairs, n_src)

    # Otherwise we want 3000 mixtures
    else:
        while len(utt_pairs) < 3000:
            utt_pairs = set_utt_pairs(aishell1_md_file, utt_pairs, n_src)
            utt_pairs = remove_duplicates(utt_pairs)
        utt_pairs = utt_pairs[:3000]

    return utt_pairs

def remove_duplicates(utt_pairs):
    print('Removing duplicates')
    # look for identical mixtures O(n²)
    for i, pair in enumerate(utt_pairs):
        for j, du_pair in enumerate(utt_pairs):
            # sort because [s1,s2] = [s2,s1]
            if sorted(pair) == sorted(du_pair) and i != j:
                utt_pairs.remove(du_pair)
    return utt_pairs

def set_utt_pairs(aishell1_md_file, pair_list, n_src):
    # A counter
    c = 0
    # Index of the rows in the metadata file
    index = list(range(len(aishell1_md_file)))

    # Try to create pairs with different speakers end after 1000 fails
    while len(index) >= n_src and c < 1000:
        couple = random.sample(index, n_src)
        # Check that speakers are different
        speaker_list = set([aishell1_md_file.iloc[couple[i]]['speaker_ID']
                            for i in range(n_src)])
        # If there are duplicates then increment the counter
        if len(speaker_list) != n_src:
            c += 1
        # Else append the combination to pair_list and erase the combination
        # from the available indexes
        else:
            for i in range(n_src):
                index.remove(couple[i])
            pair_list.append(couple)
            c = 0
    return pair_list

def check_already_generated(md_dir, dataset, to_be_ignored, aishell1_md_files):
    already_generated = os.listdir(md_dir)
    for generated in already_generated:
        if generated.startswith(f"{dataset}") and 'info' not in generated:
            if 'dev' in generated:
                to_be_ignored.append('dev.csv')
            elif 'test' in generated:
                to_be_ignored.append('test.csv')
            elif 'train' in generated:
                to_be_ignored.append('train.csv')
            print(f"{generated} already exists in "
                  f"{md_dir} it won't be overwritten")
    for element in to_be_ignored:
        aishell1_md_files.remove(element)

    return aishell1_md_files

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)