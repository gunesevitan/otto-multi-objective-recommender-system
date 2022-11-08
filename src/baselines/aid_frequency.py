import sys
import logging
import pathlib
from tqdm import tqdm
import json
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    df_test = pd.read_pickle(settings.DATA / 'test.pkl')
    logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    with open(settings.DATA / 'statistics' / 'all_20_most_frequent_click_aids.json') as f:
        all_20_most_frequent_click_aids = json.load(f)

    with open(settings.DATA / 'statistics' / 'all_20_most_frequent_cart_aids.json') as f:
        all_20_most_frequent_cart_aids = json.load(f)

    with open(settings.DATA / 'statistics' / 'all_20_most_frequent_order_aids.json') as f:
        all_20_most_frequent_order_aids = json.load(f)

    # Create a directory for saving submission file
    submissions_directory = pathlib.Path(settings.DATA / 'submissions')
    submissions_directory.mkdir(parents=True, exist_ok=True)

    df_test_session_aid_frequencies = df_test.groupby(['session', 'aid'])['aid'].count()
    # Sort values inside groups
    df_test_session_aid_frequencies = df_test_session_aid_frequencies.sort_values(ascending=False).sort_index(level='session', sort_remaining=False)
    df_test_session_aid_frequencies = df_test_session_aid_frequencies.rename('count').reset_index()
    # Create a dictionary of session id keys and list of top 20 most frequent aid values
    df_test_session_aid_frequencies = df_test_session_aid_frequencies.groupby('session')['aid'].agg(lambda x: list(x)[:20]).to_dict()

    submission = []

    for session_id, aids in tqdm(df_test_session_aid_frequencies.items()):

        for event_type in ['click', 'cart', 'order']:

            predictions = aids.copy()

            if event_type == 'click':
                predictions += list(all_20_most_frequent_click_aids.keys())[:20 - len(aids)]
            elif event_type == 'cart':
                predictions += list(all_20_most_frequent_cart_aids.keys())[:20 - len(aids)]
            elif event_type == 'order':
                predictions += list(all_20_most_frequent_order_aids.keys())[:20 - len(aids)]

            predictions = ' '.join([str(aid) for aid in predictions])
            submission.append({
                'session_type': f'{session_id}_{event_type}s',
                'labels': predictions
            })

    df_submission = pd.DataFrame(submission)
    df_submission.to_csv(submissions_directory / 'aid_frequency_submission.csv', index=False)
