import sys
import logging
import argparse
import pathlib
import yaml
import pandas as pd
import polars as pl
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import fasttext

sys.path.append('..')
import settings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    config = yaml.load(open(settings.MODELS / args.config_path, 'r'), Loader=yaml.FullLoader)

    # Create directory for models
    model_root_directory = pathlib.Path(settings.MODELS / config['persistence']['model_directory'])
    model_root_directory.mkdir(parents=True, exist_ok=True)

    if config['model']['model_name'] == 'Word2Vec':

        # Create list of sessions as sentences
        df = pd.concat((
            pd.read_pickle(settings.DATA / 'train.pkl'),
            pd.read_pickle(settings.DATA / 'test.pkl')
        ), axis=0, ignore_index=True)
        logging.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')
        sentences = pl.DataFrame(df).groupby('session').agg(pl.col('aid').alias('sentence'))['sentence'].to_list()
        del df
        n_sentences = len(sentences)
        logging.info(f'Sentences Dataset - sentences: {n_sentences}')

        model = Word2Vec(sentences=sentences, **config['model']['model_args'])
        model.save(str(model_root_directory / 'word2vec.model'))
        logging.info(f'Word2vec model finished training and saved to {model_root_directory}')

    elif config['model']['model_name'] == 'Doc2Vec':

        # Create list of sessions as documents
        df = pd.concat((
            pd.read_pickle(settings.DATA / 'train.pkl'),
            pd.read_pickle(settings.DATA / 'test.pkl')
        ), axis=0, ignore_index=True)
        logging.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')
        documents = pl.DataFrame(df).groupby('session').agg(pl.col('aid').alias('sentence'))['sentence'].to_list()
        documents = [TaggedDocument(document, [document_idx]) for document_idx, document in enumerate(documents)]
        del df
        n_documents = len(documents)
        logging.info(f'Documents Dataset - documents: {n_documents}')

        model = Doc2Vec(documents=documents, **config['model']['model_args'])
        model.save(str(model_root_directory / 'doc2vec.model'))
        logging.info(f'Doc2vec model finished training and saved to {model_root_directory}')

    elif config['model']['model_name'] == 'FastText':

        model = fasttext.train_unsupervised(input=str(settings.DATA / 'word2vec' / 'sentences.txt'), **config['model']['model_args'])
        model.save_model(str(model_root_directory / 'fasttext.bin'))
        logging.info(f'FastText model finished training and saved to {model_root_directory}')

    else:
        raise ValueError('Invalid model_name')
