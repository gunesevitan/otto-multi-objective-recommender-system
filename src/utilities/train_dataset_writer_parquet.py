import sys
import logging
import pathlib
from tqdm import tqdm
import pandas as pd

sys.path.append('..')
import settings

if __name__ == '__main__':

    df_train = pd.read_pickle(settings.DATA / 'train.pkl')
    # Last week of the training data is validation set
    df_validation_sessions = df_train.loc[df_train['session'] >= 11098528]
    # Session truncation indexes are located in train labels
    df_train_labels = pd.read_pickle(settings.DATA / 'train_labels.pkl')

    logging.info(f'Dataset Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    logging.info(f'Train Labels Shape: {df_train_labels.shape} - Memory Usage: {df_train_labels.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Iteratively truncate each session from their cut-off points take their first N + 1 events
    truncated_sessions = []

    for session, df_session  in tqdm(df_validation_sessions.groupby('session'), total=df_validation_sessions['session'].nunique()):

        session_cutoff_idx = df_train_labels.loc[df_train_labels['session'] == session, 'session_cutoff_idx'].values[0]

        df_session = df_session.head(session_cutoff_idx + 1)
        truncated_sessions.append(df_session)

    df_truncated_sessions = pd.concat(truncated_sessions, axis=0, ignore_index=True).reset_index(drop=True)
    del truncated_sessions

    # First 3 weeks of training set is concatenated with truncated last week
    df_train_sessions = df_train.loc[df_train['session'] < 11098528]
    df_train_truncated = pd.concat((df_train_sessions, df_truncated_sessions), axis=0, ignore_index=True).reset_index(drop=True)

    # Create a directory for saving parquet files
    parquet_directory = pathlib.Path(settings.DATA / 'parquet_files' / 'train_truncated_parquet')
    parquet_directory.mkdir(parents=True, exist_ok=True)

    session_chunk_size = 100_000

    for chunk_idx in range((df_train_truncated['session'].nunique() // session_chunk_size) + 1):

        session_chunk_start = chunk_idx * session_chunk_size
        session_chunk_end = (chunk_idx + 1) * session_chunk_size

        df_parquet = df_train_truncated.loc[(df_train_truncated['session'] >= session_chunk_start) & (df_train_truncated['session'] < session_chunk_end)]
        df_parquet.to_parquet(parquet_directory / f'train_truncated_{chunk_idx}.parquet')

    logging.info(f'train_truncated_chunk_idx.parquet files are saved to {parquet_directory}')
