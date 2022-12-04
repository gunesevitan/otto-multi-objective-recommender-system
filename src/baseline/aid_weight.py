import sys
import logging
import argparse
import pathlib
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings
import metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    if args.mode == 'validation':

        logging.info('Running aid weight model in validation mode')
        df_train_labels = pd.read_pickle(settings.DATA / 'train_labels.pkl')
        logging.info(f'Train Labels Shape: {df_train_labels.shape} - Memory Usage: {df_train_labels.memory_usage().sum() / 1024 ** 2:.2f} MB')

        df_train_labels['click_predictions'] = np.nan
        df_train_labels['click_predictions'] = df_train_labels['click_predictions'].astype(object)
        df_train_labels['cart_predictions'] = np.nan
        df_train_labels['cart_predictions'] = df_train_labels['cart_predictions'].astype(object)
        df_train_labels['order_predictions'] = np.nan
        df_train_labels['order_predictions'] = df_train_labels['order_predictions'].astype(object)

        event_type_coefficient = {0: 1, 1: 6, 2: 3}

        for idx, row in tqdm(df_train_labels.iterrows(), total=df_train_labels.shape[0]):

            session_aids = row['aid'][:row['session_cutoff_idx'] + 1]
            session_event_types = row['type'][:row['session_cutoff_idx'] + 1]

            weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
            session_aid_weights = defaultdict(lambda: 0)
            for aid, event_type, weight in zip(session_aids, session_event_types, weights):
                session_aid_weights[aid] += weight * event_type_coefficient[event_type]

            sorted_aids = [aid for aid, weight in sorted(session_aid_weights.items(), key=lambda item: -item[1])][:20]

            df_train_labels.at[idx, 'click_predictions'] = sorted_aids
            df_train_labels.at[idx, 'cart_predictions'] = sorted_aids
            df_train_labels.at[idx, 'order_predictions'] = sorted_aids

            df_train_labels.at[idx, 'click_recall'] = metrics.click_recall(row['click_labels'], sorted_aids)
            df_train_labels.at[idx, 'cart_recall'] = metrics.cart_order_recall(row['cart_labels'], sorted_aids)
            df_train_labels.at[idx, 'order_recall'] = metrics.cart_order_recall(row['order_labels'], sorted_aids)

        df_train_labels['recall'] = (df_train_labels['click_recall'] * 0.1) + (df_train_labels['cart_recall'] * 0.3) + (df_train_labels['order_recall'] * 0.6)
        mean_click_recall = df_train_labels['click_recall'].mean()
        mean_cart_recall = df_train_labels['cart_recall'].mean()
        mean_order_recall = df_train_labels['order_recall'].mean()
        mean_weighted_recall = df_train_labels['recall'].mean()

        logging.info(
            f'''
            aid weight model validation scores
            clicks - n: {df_train_labels["click_recall"].notna().sum()} recall@20: {mean_click_recall:.4f}
            carts - n: {df_train_labels["cart_recall"].notna().sum()} recall@20: {mean_cart_recall:.4f}
            orders - n: {df_train_labels["order_recall"].notna().sum()} recall@20: {mean_order_recall:.4f}
            weighted recall@20: {mean_weighted_recall:.4f}
            '''
        )

    elif args.mode == 'submission':

        logging.info('Running aid weight model in submission mode')
        df_test = pd.read_pickle(settings.DATA / 'test.pkl')
        logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

        # Create a directory for saving submission file
        submissions_directory = pathlib.Path(settings.DATA / 'submissions')
        submissions_directory.mkdir(parents=True, exist_ok=True)

        event_type_coefficient = {0: 1, 1: 6, 2: 3}
        df_test = df_test.groupby('session')[['aid', 'type']].agg(list).reset_index()
        test_predictions = []

        for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):

            session_aids = row['aid']
            session_event_types = row['type']

            weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
            session_aid_weights = defaultdict(lambda: 0)
            for aid, event_type, weight in zip(session_aids, session_event_types, weights):
                session_aid_weights[aid] += weight * event_type_coefficient[event_type]

            sorted_aids = [aid for aid, weight in sorted(session_aid_weights.items(), key=lambda item: -item[1])][:20]

            predictions = ' '.join([str(aid) for aid in sorted_aids])
            for event_type in ['click', 'cart', 'order']:
                test_predictions.append({
                    'session_type': f'{row["session"]}_{event_type}s',
                    'labels': predictions
                })

        df_test_predictions = pd.DataFrame(test_predictions)
        df_test_predictions.to_csv(submissions_directory / 'aid_weight_submission.csv.gz', index=False, compression='gzip')
