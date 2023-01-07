import sys
import logging
import argparse
import pathlib
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    candidate_directory = pathlib.Path(settings.DATA / 'candidate')
    candidate_directory.mkdir(parents=True, exist_ok=True)

    event_type_coefficient = {0: 1, 1: 6, 2: 1}

    if args.mode == 'validation':

        logging.info('Running recency weighted candidate generation on training set')
        df = pd.read_pickle(settings.DATA / 'train_labels.pkl')
        logging.info(f'Train Labels Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

        # Cut session aids and event types from their cutoff index
        df['aid'] = df[['aid', 'session_cutoff_idx']].apply(lambda x: x['aid'][:x['session_cutoff_idx'] + 1], axis=1)
        df['type'] = df[['type', 'session_cutoff_idx']].apply(lambda x: x['type'][:x['session_cutoff_idx'] + 1], axis=1)

        df['click_candidates'] = np.nan
        df['click_candidates'] = df['click_candidates'].astype(object)
        df['cart_candidates'] = np.nan
        df['cart_candidates'] = df['cart_candidates'].astype(object)
        df['order_candidates'] = np.nan
        df['order_candidates'] = df['order_candidates'].astype(object)

        df['click_candidate_scores'] = np.nan
        df['click_candidate_scores'] = df['click_candidate_scores'].astype(object)
        df['cart_candidate_scores'] = np.nan
        df['cart_candidate_scores'] = df['cart_candidate_scores'].astype(object)
        df['order_candidate_scores'] = np.nan
        df['order_candidate_scores'] = df['order_candidate_scores'].astype(object)

        df['click_candidate_labels'] = np.nan
        df['click_candidate_labels'] = df['click_candidate_labels'].astype(object)
        df['cart_candidate_labels'] = np.nan
        df['cart_candidate_labels'] = df['cart_candidate_labels'].astype(object)
        df['order_candidate_labels'] = np.nan
        df['order_candidate_labels'] = df['order_candidate_labels'].astype(object)

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

            session_aids = row['aid']
            session_event_types = row['type']
            session_unique_aids = list(dict.fromkeys(session_aids[::-1]))

            # Calculate click, cart and order weights based on recency
            click_recency_weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
            cart_recency_weights = np.logspace(0.5, 1, len(session_aids), base=2, endpoint=True) - 1
            order_recency_weights = np.logspace(0.5, 1, len(session_aids), base=2, endpoint=True) - 1
            session_aid_click_weights = Counter()
            session_aid_cart_weights = Counter()
            session_aid_order_weights = Counter()

            for aid, event_type, click_recency_weight, cart_recency_weight, order_recency_weight in zip(session_aids, session_event_types, click_recency_weights, cart_recency_weights, order_recency_weights):
                session_aid_click_weights[aid] += (click_recency_weight * event_type_coefficient[event_type])
                session_aid_cart_weights[aid] += (cart_recency_weight * event_type_coefficient[event_type])
                session_aid_order_weights[aid] += (order_recency_weight * event_type_coefficient[event_type])

            # Sort aids by their weights in descending order
            sorted_click_aids = [aid for aid, weight in session_aid_click_weights.most_common(len(session_unique_aids))]
            sorted_cart_aids = [aid for aid, weight in session_aid_cart_weights.most_common(len(session_unique_aids))]
            sorted_order_aids = [aid for aid, weight in session_aid_order_weights.most_common(len(session_unique_aids))]

            df.at[idx, 'click_candidates'] = sorted_click_aids
            df.at[idx, 'cart_candidates'] = sorted_cart_aids
            df.at[idx, 'order_candidates'] = sorted_order_aids

            # Sort aids by their weights in descending order
            sorted_click_aid_weights = [weight for aid, weight in session_aid_click_weights.most_common(len(session_unique_aids))]
            sorted_cart_aid_weights = [weight for aid, weight in session_aid_cart_weights.most_common(len(session_unique_aids))]
            sorted_order_aid_weights = [weight for aid, weight in session_aid_order_weights.most_common(len(session_unique_aids))]

            df.at[idx, 'click_candidate_scores'] = sorted_click_aid_weights
            df.at[idx, 'cart_candidate_scores'] = sorted_cart_aid_weights
            df.at[idx, 'order_candidate_scores'] = sorted_order_aid_weights

            # Create candidate labels for clicks, carts and orders
            sorted_click_aid_labels = [int(aid == row['click_labels']) for aid in sorted_click_aids]
            sorted_cart_aid_labels = [int(aid in row['cart_labels']) for aid in sorted_cart_aids]
            sorted_order_aid_labels = [int(aid in row['order_labels']) for aid in sorted_order_aids]

            df.at[idx, 'click_candidate_labels'] = sorted_click_aid_labels
            df.at[idx, 'cart_candidate_labels'] = sorted_cart_aid_labels
            df.at[idx, 'order_candidate_labels'] = sorted_order_aid_labels

        candidate_columns = [
            'click_candidates', 'cart_candidates', 'order_candidates',
            'click_candidate_scores', 'cart_candidate_scores', 'order_candidate_scores',
            'click_candidate_labels', 'cart_candidate_labels', 'order_candidate_labels'
        ]
        df = df.explode(candidate_columns)[['session'] + candidate_columns]
        df['click_candidates'] = df['click_candidates'].astype(np.uint64)
        df['cart_candidates'] = df['cart_candidates'].astype(np.uint64)
        df['order_candidates'] = df['order_candidates'].astype(np.uint64)
        df['click_candidate_scores'] = df['click_candidate_scores'].astype(np.float32)
        df['cart_candidate_scores'] = df['cart_candidate_scores'].astype(np.float32)
        df['order_candidate_scores'] = df['order_candidate_scores'].astype(np.float32)
        df['click_candidate_labels'] = df['click_candidate_labels'].astype(np.uint8)
        df['cart_candidate_labels'] = df['cart_candidate_labels'].astype(np.uint8)
        df['order_candidate_labels'] = df['order_candidate_labels'].astype(np.uint8)
        df.to_pickle(candidate_directory / 'recency_weighted_validation.pkl')

        logging.info(
            f'''
            Recency weighting candidate generation
            {df.shape[0]} candidates are generated for {df["session"].nunique()} sessions
            Candidate labels
            Click candidates - Positives: {df["click_candidate_labels"].sum()} Negatives: {(df["click_candidate_labels"] == 0).sum()}
            Cart candidates - Positives: {df["cart_candidate_labels"].sum()} Negatives: {(df["cart_candidate_labels"] == 0).sum()}
            Order candidates - Positives: {df["order_candidate_labels"].sum()} Negatives: {(df["order_candidate_labels"] == 0).sum()}
            Candidate scores
            Click candidates - Positives {df.loc[df["click_candidate_labels"] == 1, "click_candidate_scores"].mean():.4f} Negatives {df.loc[df["click_candidate_labels"] == 0, "click_candidate_scores"].mean():.4f} All {df["click_candidate_scores"].mean():.4f}
            Cart candidates - Positives {df.loc[df["cart_candidate_labels"] == 1, "cart_candidate_scores"].mean():.4f} Negatives {df.loc[df["cart_candidate_labels"] == 0, "cart_candidate_scores"].mean():.4f} All {df["cart_candidate_scores"].mean():.4f}
            Order candidates - Positives {df.loc[df["order_candidate_labels"] == 1, "order_candidate_scores"].mean():.4f} Negatives {df.loc[df["order_candidate_labels"] == 0, "order_candidate_scores"].mean():.4f} All {df["order_candidate_scores"].mean():.4f}
            '''
        )

    elif args.mode == 'test':

        logging.info('Running recency weighted candidate generation on test set')
        df = pd.read_pickle(settings.DATA / 'test.pkl')
        df = df.groupby('session')[['aid', 'type']].agg(list).reset_index()
        logging.info(f'Test Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

        df['click_candidates'] = np.nan
        df['click_candidates'] = df['click_candidates'].astype(object)
        df['cart_candidates'] = np.nan
        df['cart_candidates'] = df['cart_candidates'].astype(object)
        df['order_candidates'] = np.nan
        df['order_candidates'] = df['order_candidates'].astype(object)

        df['click_candidate_scores'] = np.nan
        df['click_candidate_scores'] = df['click_candidate_scores'].astype(object)
        df['cart_candidate_scores'] = np.nan
        df['cart_candidate_scores'] = df['cart_candidate_scores'].astype(object)
        df['order_candidate_scores'] = np.nan
        df['order_candidate_scores'] = df['order_candidate_scores'].astype(object)

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

            session_aids = row['aid']
            session_event_types = row['type']
            session_unique_aids = list(dict.fromkeys(session_aids[::-1]))

            # Calculate click, cart and order weights based on recency
            click_recency_weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
            cart_recency_weights = np.logspace(0.5, 1, len(session_aids), base=2, endpoint=True) - 1
            order_recency_weights = np.logspace(0.5, 1, len(session_aids), base=2, endpoint=True) - 1
            session_aid_click_weights = Counter()
            session_aid_cart_weights = Counter()
            session_aid_order_weights = Counter()

            for aid, event_type, click_recency_weight, cart_recency_weight, order_recency_weight in zip(session_aids, session_event_types, click_recency_weights, cart_recency_weights, order_recency_weights):
                session_aid_click_weights[aid] += (click_recency_weight * event_type_coefficient[event_type])
                session_aid_cart_weights[aid] += (cart_recency_weight * event_type_coefficient[event_type])
                session_aid_order_weights[aid] += (order_recency_weight * event_type_coefficient[event_type])

            # Sort aids by their weights in descending order
            sorted_click_aids = [aid for aid, weight in session_aid_click_weights.most_common(len(session_unique_aids))]
            sorted_cart_aids = [aid for aid, weight in session_aid_cart_weights.most_common(len(session_unique_aids))]
            sorted_order_aids = [aid for aid, weight in session_aid_order_weights.most_common(len(session_unique_aids))]

            df.at[idx, 'click_candidates'] = sorted_click_aids
            df.at[idx, 'cart_candidates'] = sorted_cart_aids
            df.at[idx, 'order_candidates'] = sorted_order_aids

            # Sort aids by their weights in descending order
            sorted_click_aid_weights = [weight for aid, weight in session_aid_click_weights.most_common(len(session_unique_aids))]
            sorted_cart_aid_weights = [weight for aid, weight in session_aid_cart_weights.most_common(len(session_unique_aids))]
            sorted_order_aid_weights = [weight for aid, weight in session_aid_order_weights.most_common(len(session_unique_aids))]

            df.at[idx, 'click_candidate_scores'] = sorted_click_aid_weights
            df.at[idx, 'cart_candidate_scores'] = sorted_cart_aid_weights
            df.at[idx, 'order_candidate_scores'] = sorted_order_aid_weights

        candidate_columns = [
            'click_candidates', 'cart_candidates', 'order_candidates',
            'click_candidate_scores', 'cart_candidate_scores', 'order_candidate_scores'
        ]
        df = df.explode(candidate_columns)[['session'] + candidate_columns]
        df['click_candidates'] = df['click_candidates'].astype(np.uint64)
        df['cart_candidates'] = df['cart_candidates'].astype(np.uint64)
        df['order_candidates'] = df['order_candidates'].astype(np.uint64)
        df['click_candidate_scores'] = df['click_candidate_scores'].astype(np.float32)
        df['cart_candidate_scores'] = df['cart_candidate_scores'].astype(np.float32)
        df['order_candidate_scores'] = df['order_candidate_scores'].astype(np.float32)
        df.to_pickle(candidate_directory / 'recency_weighted_test.pkl')

        logging.info(
            f'''
            Recency weighting candidate generation
            {df.shape[0]} candidates are generated for {df["session"].nunique()} sessions
            Candidate scores
            Clicks - Average score: {df["click_candidate_scores"].mean():.4f}
            Carts - Average Score: {df["cart_candidate_scores"].mean():.4f}
            Orders - Average Score {df["order_candidate_scores"].mean():.4f}
            '''
        )

    else:
        raise ValueError('Invalid mode')
