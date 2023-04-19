#!/bin/bash

aishell1_dir=/data/shared/speech/AISHELL1
aishell1_md_dir=/home/huangpeng/code/LibriMix/metadata/AISHELL1
metadata_outdir=/home/huangpeng/code/LibriMix/metadata/AISHELL1_2mix
n_src=2
aishell1mix_outdir=/data/shared/speech/


# python scripts/create_aishell1mix_metadata.py --aishell1_dir $aishell1_dir --aishell1_md_dir $aishell1_md_dir --metadata_outdir $metadata_outdir --n_src $n_src

python scripts/create_aishell1mix_from_metadata.py --aishell1_dir $aishell1_dir --metadata_dir $metadata_outdir --aishell1mix_outdir $aishell1mix_outdir --n_src $n_src --freqs '8k' --modes 'max' --types 'mix_clean'

