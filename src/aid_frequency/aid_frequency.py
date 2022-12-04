import sys
import logging
import argparse
import pathlib
from collections import Counter
from tqdm import tqdm
import json
import numpy as np
import pandas as pd

sys.path.append('..')
import settings
import metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    with open(settings.DATA / 'aid_frequencies' / 'all_20_most_frequent_click_aids.json') as f:
        all_20_most_frequent_click_aids = list(json.load(f).keys())

    with open(settings.DATA / 'aid_frequencies' / 'all_20_most_frequent_cart_aids.json') as f:
        all_20_most_frequent_cart_aids = list(json.load(f).keys())

    with open(settings.DATA / 'aid_frequencies' / 'all_20_most_frequent_order_aids.json') as f:
        all_20_most_frequent_order_aids = list(json.load(f).keys())

    if args.mode == 'validation':

        logging.info('Running aid frequency model in validation mode')
        df_train_labels = pd.read_pickle(settings.DATA / 'train_labels.pkl')
        logging.info(f'Train Labels Shape: {df_train_labels.shape} - Memory Usage: {df_train_labels.memory_usage().sum() / 1024 ** 2:.2f} MB')

        df_train_labels['click_predictions'] = np.nan
        df_train_labels['click_predictions'] = df_train_labels['click_predictions'].astype(object)
        df_train_labels['cart_predictions'] = np.nan
        df_train_labels['cart_predictions'] = df_train_labels['cart_predictions'].astype(object)
        df_train_labels['order_predictions'] = np.nan
        df_train_labels['order_predictions'] = df_train_labels['order_predictions'].astype(object)

        for idx, row in tqdm(df_train_labels.iterrows(), total=df_train_labels.shape[0]):

            session_most_frequent_aids = list(Counter(row['aid'][:row['session_cutoff_idx'] + 1]).keys())[:20]

            click_predictions = session_most_frequent_aids + all_20_most_frequent_click_aids[:20 - len(session_most_frequent_aids)]
            cart_predictions = session_most_frequent_aids + all_20_most_frequent_cart_aids[:20 - len(session_most_frequent_aids)]
            order_predictions = session_most_frequent_aids + all_20_most_frequent_order_aids[:20 - len(session_most_frequent_aids)]

            df_train_labels.at[idx, 'click_predictions'] = click_predictions
            df_train_labels.at[idx, 'cart_predictions'] = cart_predictions
            df_train_labels.at[idx, 'order_predictions'] = order_predictions

            df_train_labels.at[idx, 'click_recall'] = metrics.click_recall(row['click_labels'], click_predictions)
            df_train_labels.at[idx, 'cart_recall'] = metrics.cart_order_recall(row['cart_labels'], cart_predictions)
            df_train_labels.at[idx, 'order_recall'] = metrics.cart_order_recall(row['order_labels'], order_predictions)

        df_train_labels['recall'] = (df_train_labels['click_recall'] * 0.1) + (df_train_labels['cart_recall'] * 0.3) + (df_train_labels['order_recall'] * 0.6)
        mean_click_recall = df_train_labels['click_recall'].mean()
        mean_cart_recall = df_train_labels['cart_recall'].mean()
        mean_order_recall = df_train_labels['order_recall'].mean()
        mean_weighted_recall = df_train_labels['recall'].mean()

        logging.info(
            f'''
            aid frequency model validation scores
            clicks - n: {df_train_labels["click_recall"].notna().sum()} recall@20: {mean_click_recall:.4f}
            carts - n: {df_train_labels["cart_recall"].notna().sum()} recall@20: {mean_cart_recall:.4f}
            orders - n: {df_train_labels["order_recall"].notna().sum()} recall@20: {mean_order_recall:.4f}
            weighted recall@20: {mean_weighted_recall:.4f}
            '''
        )

    elif args.mode == 'submission':

        logging.info('Running aid frequency model in submission mode')
        df_test = pd.read_pickle(settings.DATA / 'test.pkl')
        logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

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
        df_submission.to_csv(submissions_directory / 'aid_frequency_submission.csv.gz', index=False, compression='gzip')
