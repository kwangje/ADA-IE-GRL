import pandas as pd
from pathlib import Path
import numpy as np
import os
import argparse


def make_vox1_df(root_path: Path) -> pd.DataFrame:
    all_files = list((root_path/'wav').rglob('**/*.wav'))
    speakers = ['vox1-' + f.parents[1].stem for f in all_files]
    df = pd.DataFrame({'path': all_files, 'speaker': speakers})
    return df

def make_vox2_df(root_path: Path) -> pd.DataFrame:
    all_files = list((root_path/'dev/wav').rglob('**/*.wav'))
    speakers = ['vox2-' + f.parents[1].stem for f in all_files]
    df = pd.DataFrame({'path': all_files, 'speaker': speakers})
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate train & valid csvs from dataset directories")

    parser.add_argument('--vox1_path', default=None, type=str)
    parser.add_argument('--vox2_path', default=None, type=str)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--valid_spks', default=200, type=int)

    args = parser.parse_args()

    cat_dfs = []
    if args.vox1_path is not None:
        vox1_df = make_vox1_df(Path(args.vox1_path))
        cat_dfs.append(vox1_df)
    if args.vox2_path is not None:
        vox2_df = make_vox2_df(Path(args.vox2_path))
        cat_dfs.append(vox2_df)

    full_df = pd.concat(cat_dfs)
    print("Preliminary number of speakers: ", len(full_df.speaker.unique()))
    full_df['cnt'] = full_df.groupby('speaker')['path'].transform('count')
    full_df = full_df[full_df.cnt > 8] 
    full_df.drop('cnt', axis=1)

    speakers = full_df.speaker.unique()
    np.random.seed(args.seed)
    np.random.shuffle(speakers)
    n_valid = args.valid_spks
    train_spks = speakers[:-n_valid]
    valid_spks = speakers[-n_valid:]
    train_df = full_df[full_df.speaker.isin(train_spks)]
    print(f"Finished constructing train df of {len(train_df):,d} utterances.")
    valid_df = full_df[full_df.speaker.isin(valid_spks)]
    print(f"Finished constructing valid df of {len(valid_df):,d} utterances ({len(valid_df.speaker.unique())} speakers).")
    os.makedirs('splits', exist_ok=True)
    train_df.to_csv("splits/train.csv.zip", index=False, compression='zip')
    valid_df.to_csv("splits/valid.csv.zip", index=False, compression='zip')

if __name__ == '__main__':
    main()