import sys
import logging
import pathlib
from tqdm import tqdm
import pandas as pd

sys.path.append('..')
import settings

if __name__ == '__main__':


    df = pd.concat((
        pd.read_parquet(settings.DATA / 'splits' / 'train.parquet'),
        pd.read_parquet(settings.DATA / 'splits' / 'val.parquet')
    ), axis=0, ignore_index=True).reset_index(drop=True)
    df.sort_values(by=['session', 'ts'], ascending=[True, True], inplace=True)

    logging.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Create a directory for saving parquet files
    parquet_directory = pathlib.Path(settings.DATA / 'parquet_files' / 'train_truncated_parquet')
    parquet_directory.mkdir(parents=True, exist_ok=True)

    session_chunk_size = 100_000

    for chunk_idx in tqdm(range((df['session'].nunique() // session_chunk_size) + 1)):

        session_chunk_start = chunk_idx * session_chunk_size
        session_chunk_end = (chunk_idx + 1) * session_chunk_size

        df_parquet = df.loc[(df['session'] >= session_chunk_start) & (df['session'] < session_chunk_end)].reset_index(drop=True)
        df_parquet.to_parquet(parquet_directory / f'train_truncated_{chunk_idx}.parquet')

    logging.info(f'train_truncated_chunk_idx.parquet files are saved to {parquet_directory}')
