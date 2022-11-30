import logging
import yaml
from tqdm import tqdm
import pandas as pd
import polars as pl
import torch
from annoy import AnnoyIndex

import settings
from matrix_factorization.torch_modules import MatrixFactorization


if __name__ == '__main__':

    matrix_factorization_config = yaml.load(open(settings.MODELS / 'matrix_factorization' / 'config.yaml', 'r'), Loader=yaml.FullLoader)

    matrix_factorization_dataset_directory = settings.DATA / matrix_factorization_config['dataset']['dataset_directory']
    df_matrix_factorization = pl.concat((
        pl.read_parquet(matrix_factorization_dataset_directory / 'train.parquet'),
        pl.read_parquet(matrix_factorization_dataset_directory / 'test.parquet')
    ))
    logging.info(f'Matrix Factorization Dataset Shape: {df_matrix_factorization.shape}')
    n_sessions = df_matrix_factorization['session'].n_unique()
    n_aids = df_matrix_factorization['aid'].n_unique()

    # Create pairs and write train/validation parquet datasets to disk
    '''matrix_factorization_pairs = df_matrix_factorization.groupby('session').agg([
        pl.col('aid'),
        pl.col('aid').shift(-1).alias('aid_next')
    ]).explode(['aid', 'aid_next']).drop_nulls()[['aid', 'aid_next']]'''
    matrix_factorization_model = MatrixFactorization(
        n_sessions=n_sessions,
        n_aids=n_aids,
        embedding_dim=matrix_factorization_config['model']['embedding_dim']
    )
    del df_matrix_factorization
    matrix_factorization_model.load_state_dict(torch.load(settings.MODELS / 'matrix_factorization' / 'model_best.pt'))
    matrix_factorization_embeddings = matrix_factorization_model.aid_embeddings.weight.detach().numpy()
    del matrix_factorization_model
    logging.info(f'matrix_factorization/model_best.pt is loaded')

    annoy_index = AnnoyIndex(8, 'euclidean')
    for embedding_idx, vector in tqdm(enumerate(matrix_factorization_embeddings), total=matrix_factorization_embeddings.shape[0]):
        annoy_index.add_item(embedding_idx, vector)
    annoy_index.build(n_trees=10, n_jobs=-1)
    logging.info(f'annoy index is built')

    df_test = pd.read_pickle(settings.DATA / 'test.pkl')
    logging.info(f'Dataset Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    submission = []

    df_test_session_aid_frequencies = df_test.groupby(['session', 'aid'])['aid'].count()
    # Sort values inside groups
    df_test_session_aid_frequencies = df_test_session_aid_frequencies.sort_values(ascending=False).sort_index(level='session', sort_remaining=False)
    df_test_session_aid_frequencies = df_test_session_aid_frequencies.rename('count').reset_index()
    # Create a dictionary of session id keys and list of top 20 most frequent aid values
    df_test_session_aid_frequencies = df_test_session_aid_frequencies.groupby('session')['aid'].agg(lambda x: list(x)[:20]).to_dict()

    for session_id, aids in tqdm(df_test_session_aid_frequencies.items()):

        for event_type in ['click', 'cart', 'order']:

            predictions = aids.copy()
            predictions += annoy_index.get_nns_by_item(predictions[-1], 21 - len(predictions))[1:]
            predictions = ' '.join([str(aid) for aid in predictions])
            submission.append({
                'session_type': f'{session_id}_{event_type}s',
                'labels': predictions
            })

    submission = pd.DataFrame(submission)
    submission.to_csv('submission.csv', index=False)