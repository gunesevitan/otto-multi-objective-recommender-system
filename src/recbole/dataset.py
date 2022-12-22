import sys
import logging
import pathlib
import pandas as pd
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    # Use sessions after the cutoff point
    SESSION_CUTOFF = 10000000

    df = pd.concat((
        pd.read_pickle(settings.DATA / 'train.pkl'),
        pd.read_pickle(settings.DATA / 'test.pkl')
    ), axis=0, ignore_index=True)
    df = df.loc[df['session'] >= SESSION_CUTOFF]
    logging.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df['ts'] = df['ts'].view(int)
    df = pl.DataFrame(df)
    df = df.sort(['session', 'aid', 'ts'])
    df = df.with_columns((pl.col('ts') * 1e9).alias('ts'))
    df = df.rename({'session': 'session:token', 'aid': 'aid:token', 'ts': 'ts:float'})

    # Create directory for model specific dataset
    dataset_root_directory = pathlib.Path(settings.DATA / 'recbole')
    dataset_root_directory.mkdir(parents=True, exist_ok=True)

    df['session:token', 'aid:token', 'ts:float'].write_csv(dataset_root_directory / 'data' / 'data.inter', sep='\t')
    logging.info(f'Saved data.inter to {dataset_root_directory / "data"}')
