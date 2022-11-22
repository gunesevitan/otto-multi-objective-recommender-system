import sys
import logging
import pathlib
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd

sys.path.append('..')
import settings


def create_covisitation_counter(df, chunk_size=30000, n_last_events=30, hour_difference=24):

    """
    Create co-visitation counter from given sessions on given dataframe

    Parameters
    ----------
    df: pandas.DataFrame of shape (n_event, 4)
        Training or test dataframe with session, aid, ts, type columns

    chunk_size: int
        Chunk size of sessions

    n_last_events: int
        Number of last events per session

    hour_difference: int
        Maximum hour difference between aid pairs

    Returns
    -------
    covisitation_counter: defaultdict with Counter default type
        A dictionary of Counter objects with co-visitation values
    """

    covisitation_counter = defaultdict(Counter)
    sessions = df['session'].unique()

    for i in tqdm(range(0, sessions.shape[0], chunk_size)):

        # Index a chunk of sessions
        df_sessions = df.loc[sessions[i]:sessions[min(sessions.shape[0] - 1, i + chunk_size - 1)]].reset_index(drop=True)
        # Create groups of sessions with n last events
        df_sessions = df_sessions.groupby('session', as_index=False).nth(list(range(-n_last_events, 0))).reset_index(drop=True)
        # Merge groups of sessions with itself for vectorized aid comparison
        df_aid_pairs = df_sessions.merge(df_sessions, on='session')
        # Drop duplicate aid pairs
        df_aid_pairs = df_aid_pairs.loc[df_aid_pairs['aid_x'] != df_aid_pairs['aid_y'], :]
        # Drop aid pairs with more than specified hour difference
        df_aid_pairs['hours_elapsed'] = (df_aid_pairs['ts_y'] - df_aid_pairs['ts_x']).dt.seconds / 3600
        df_aid_pairs = df_aid_pairs.loc[(df_aid_pairs['hours_elapsed'] > 0) & (df_aid_pairs['hours_elapsed'] <= hour_difference)]

        df_aid_pairs.drop_duplicates(['session', 'aid_x', 'aid_y'], inplace=True)
        for aid_x, aid_y in zip(df_aid_pairs['aid_x'], df_aid_pairs['aid_y']):
            covisitation_counter[aid_x][aid_y] += 1

    return covisitation_counter


if __name__ == '__main__':

    df = pd.concat((
        pd.read_pickle(settings.DATA / 'train.pkl'),
        pd.read_pickle(settings.DATA / 'test.pkl')
    ), axis=0, ignore_index=True)
    df.index = pd.MultiIndex.from_frame(df[['session']])
    logging.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Create a directory for saving extracted statistics
    covisitation_directory = pathlib.Path(settings.DATA / 'covisitation')
    covisitation_directory.mkdir(parents=True, exist_ok=True)

    n_last_events = 30
    hour_difference = 24
    covisitation_counter = create_covisitation_counter(
        df,
        chunk_size=30000,
        n_last_events=n_last_events,
        hour_difference=hour_difference
    )

    with open(covisitation_directory / f'covisitation_counter_{n_last_events}_{hour_difference}.pkl', 'wb') as f:
        pickle.dump(covisitation_counter, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f'Saved covisitation_counter_{n_last_events}_{hour_difference}.pkl to {covisitation_directory}')
