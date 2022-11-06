import sys
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


def create_dataframe(json_file_path, chunk_size):

    """
    Create pandas.DataFrame from given json_file_path

    Parameters
    ----------
    json_file_path: path-like str
        Path of the json file

    chunk_size: int
        Size of chunks while reading the json file

    Returns
    -------
    df: pandas.DataFrame of shape (n_samples, 4)
    """

    type_dict = {
        'clicks': 0,
        'carts': 1,
        'orders': 2
    }

    chunks = pd.read_json(json_file_path, lines=True, chunksize=chunk_size)
    df = pd.DataFrame()

    for chunk_idx, chunk in enumerate(chunks):

        logging.info(f'Reading Chunk {chunk_idx} ({chunk_size * chunk_idx}-{(chunk_size * (chunk_idx + 1))})')

        event_dict = {
            'session': [],
            'aid': [],
            'ts': [],
            'type': [],
        }

        for session, events in tqdm(zip(chunk['session'].tolist(), chunk['events'].tolist()), total=len(chunk['session'].tolist())):
            for event in events:
                event_dict['session'].append(session)
                event_dict['aid'].append(event['aid'])
                event_dict['ts'].append(event['ts'])
                event_dict['type'].append(type_dict[event['type']])

        chunk_session = pd.DataFrame(event_dict)
        chunk_session['session'] = chunk_session['session'].astype(np.uint32)
        chunk_session['aid'] = chunk_session['aid'].astype(np.uint32)
        chunk_session['ts'] = pd.to_datetime(chunk_session['ts'], unit='ms')
        chunk_session['type'] = chunk_session['type'].astype(np.uint8)
        df = pd.concat([df, chunk_session])

    df.reset_index(drop=True, inplace=True)

    return df


if __name__ == '__main__':

    training_json_file_path = settings.DATA / 'train.jsonl'
    test_json_file_path = settings.DATA / 'test.jsonl'

    df_train = create_dataframe(json_file_path=training_json_file_path, chunk_size=100000)
    df_test = create_dataframe(json_file_path=test_json_file_path, chunk_size=100000)

    logging.info(f'Training Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    df_train.to_pickle(settings.DATA / 'train.pkl')
    df_test.to_pickle(settings.DATA / 'test.pkl')
