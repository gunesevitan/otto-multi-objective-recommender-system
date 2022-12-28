import sys
import logging
import pathlib
from tqdm import tqdm
import pandas as pd
import polars as pl

sys.path.append('..')
import settings


if __name__ == '__main__':

    df = pd.concat((
        pd.read_pickle(settings.DATA / 'train.pkl'),
        pd.read_pickle(settings.DATA / 'test.pkl')
    ), axis=0, ignore_index=True)
    logging.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    sentences = pl.DataFrame(df).groupby('session').agg(pl.col('aid').alias('sentence'))['sentence'].to_pandas().apply(lambda x: ' '.join([str(aid) for aid in list(x)]))
    del df
    n_sentences = len(sentences)
    logging.info(f'Sentences Dataset - sentences: {n_sentences}')

    # Create directory for model specific dataset
    dataset_root_directory = pathlib.Path(settings.DATA / 'word2vec')
    dataset_root_directory.mkdir(parents=True, exist_ok=True)

    # Write every session as a line and append with \n character
    with open(dataset_root_directory / 'sentences.txt', mode='w') as f:
        for sentence in tqdm(sentences):
            f.write(sentence + '\n')
    logging.info(f'Saved sentences.txt to {dataset_root_directory}')
