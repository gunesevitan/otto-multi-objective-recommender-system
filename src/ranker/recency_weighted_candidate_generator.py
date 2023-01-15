import sys
import logging
import argparse
import pathlib
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
import polars as pl

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
        df_val = pd.read_parquet(settings.DATA / 'splits' / 'val.parquet')
        df_val = df_val.groupby('session')[['aid', 'type']].agg(list).reset_index()
        df_val_labels = pd.read_parquet(settings.DATA / 'splits' / 'val_labels.parquet')
        df_val_labels['type'] = df_val_labels['type'].map({'clicks': 0, 'carts': 1, 'orders': 2})
        df_val = df_val.merge(df_val_labels.loc[df_val_labels['type'] == 0, ['session', 'ground_truth']].rename(columns={'ground_truth': 'click_labels'}), how='left', on='session')
        df_val = df_val.merge(df_val_labels.loc[df_val_labels['type'] == 1, ['session', 'ground_truth']].rename(columns={'ground_truth': 'cart_labels'}), how='left', on='session')
        df_val = df_val.merge(df_val_labels.loc[df_val_labels['type'] == 2, ['session', 'ground_truth']].rename(columns={'ground_truth': 'order_labels'}), how='left', on='session')
        df_val = df_val.fillna(df_val.notna().applymap(lambda x: x or []))
        del df_val_labels
        logging.info(f'Validation Labels Shape: {df_val.shape} - Memory Usage: {df_val.memory_usage().sum() / 1024 ** 2:.2f} MB')

        df_val['click_candidates'] = np.nan
        df_val['click_candidates'] = df_val['click_candidates'].astype(object)
        df_val['cart_candidates'] = np.nan
        df_val['cart_candidates'] = df_val['cart_candidates'].astype(object)
        df_val['order_candidates'] = np.nan
        df_val['order_candidates'] = df_val['order_candidates'].astype(object)

        df_val['click_candidate_scores'] = np.nan
        df_val['click_candidate_scores'] = df_val['click_candidate_scores'].astype(object)
        df_val['cart_candidate_scores'] = np.nan
        df_val['cart_candidate_scores'] = df_val['cart_candidate_scores'].astype(object)
        df_val['order_candidate_scores'] = np.nan
        df_val['order_candidate_scores'] = df_val['order_candidate_scores'].astype(object)

        df_val['click_candidate_labels'] = np.nan
        df_val['click_candidate_labels'] = df_val['click_candidate_labels'].astype(object)
        df_val['cart_candidate_labels'] = np.nan
        df_val['cart_candidate_labels'] = df_val['cart_candidate_labels'].astype(object)
        df_val['order_candidate_labels'] = np.nan
        df_val['order_candidate_labels'] = df_val['order_candidate_labels'].astype(object)

        for idx, row in tqdm(df_val.iterrows(), total=df_val.shape[0]):

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

            df_val.at[idx, 'click_candidates'] = sorted_click_aids
            df_val.at[idx, 'cart_candidates'] = sorted_cart_aids
            df_val.at[idx, 'order_candidates'] = sorted_order_aids

            # Sort aids by their weights in descending order
            sorted_click_aid_weights = [weight for aid, weight in session_aid_click_weights.most_common(len(session_unique_aids))]
            sorted_cart_aid_weights = [weight for aid, weight in session_aid_cart_weights.most_common(len(session_unique_aids))]
            sorted_order_aid_weights = [weight for aid, weight in session_aid_order_weights.most_common(len(session_unique_aids))]

            df_val.at[idx, 'click_candidate_scores'] = sorted_click_aid_weights
            df_val.at[idx, 'cart_candidate_scores'] = sorted_cart_aid_weights
            df_val.at[idx, 'order_candidate_scores'] = sorted_order_aid_weights

            # Create candidate labels for clicks, carts and orders
            sorted_click_aid_labels = [int(aid == row['click_labels']) for aid in sorted_click_aids]
            sorted_cart_aid_labels = [int(aid in row['cart_labels']) for aid in sorted_cart_aids]
            sorted_order_aid_labels = [int(aid in row['order_labels']) for aid in sorted_order_aids]

            df_val.at[idx, 'click_candidate_labels'] = sorted_click_aid_labels
            df_val.at[idx, 'cart_candidate_labels'] = sorted_cart_aid_labels
            df_val.at[idx, 'order_candidate_labels'] = sorted_order_aid_labels


        df_val['click_hits'] = pl.DataFrame(df_val[['click_candidates', 'click_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        click_recall = df_val['click_hits'].sum() / df_val['click_labels'].apply(len).clip(0, 20).sum()
        df_val['cart_hits'] = pl.DataFrame(df_val[['cart_candidates', 'cart_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        cart_recall = df_val['cart_hits'].sum() / df_val['cart_labels'].apply(len).clip(0, 20).sum()
        df_val['order_hits'] = pl.DataFrame(df_val[['order_candidates', 'order_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        order_recall = df_val['order_hits'].sum() / df_val['order_labels'].apply(len).clip(0, 20).sum()
        weighted_recall = (click_recall * 0.1) + (cart_recall * 0.3) + (order_recall * 0.6)

        logging.info(
            f'''
            Candidate max recalls
            Click max recall: {click_recall:.6f}
            Cart max recall: {cart_recall:.6f}
            Order max recall: {order_recall:.6f}
            Weighted max recall: {weighted_recall:.6f}
            '''
        )

        for event_type in ['click', 'cart', 'order']:

            candidate_columns = [f'{event_type}_candidates', f'{event_type}_candidate_scores', f'{event_type}_candidate_labels']
            df_val_event = df_val.explode(candidate_columns)[['session'] + candidate_columns].rename(columns={
                f'{event_type}_candidates': 'candidates',
                f'{event_type}_candidate_scores': 'candidate_scores',
                f'{event_type}_candidate_labels': 'candidate_labels'
            })
            logging.info(
                f'''
                {event_type} recency weighted candidate generation
                {df_val_event.shape[0]} candidates are generated for {df_val_event["session"].nunique()} sessions
                Candidate labels - Positives: {df_val_event["candidate_labels"].sum()} Negatives: {(df_val_event["candidate_labels"] == 0).sum()}
                Candidate scores - Positives {df_val_event.loc[df_val_event["candidate_labels"] == 1, "candidate_scores"].mean():.4f} Negatives {df_val_event.loc[df_val_event["candidate_labels"] == 0, "candidate_scores"].mean():.4f} All {df_val_event["candidate_scores"].mean():.4f}
                '''
            )
            df_val_event['candidates'] = df_val_event['candidates'].astype(np.uint64)
            df_val_event['candidate_scores'] = df_val_event['candidate_scores'].astype(np.float32)
            df_val_event['candidate_labels'] = df_val_event['candidate_labels'].astype(np.uint8)
            df_val_event.to_pickle(candidate_directory / f'{event_type}_recency_weighted_validation.pkl')

    elif args.mode == 'submission':

        logging.info('Running recency weighted candidate generation on test set')
        df_test = pd.read_pickle(settings.DATA / 'test.pkl')
        df_test = df_test.groupby('session')[['aid', 'type']].agg(list).reset_index()
        logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

        df_test['click_candidates'] = np.nan
        df_test['click_candidates'] = df_test['click_candidates'].astype(object)
        df_test['cart_candidates'] = np.nan
        df_test['cart_candidates'] = df_test['cart_candidates'].astype(object)
        df_test['order_candidates'] = np.nan
        df_test['order_candidates'] = df_test['order_candidates'].astype(object)

        df_test['click_candidate_scores'] = np.nan
        df_test['click_candidate_scores'] = df_test['click_candidate_scores'].astype(object)
        df_test['cart_candidate_scores'] = np.nan
        df_test['cart_candidate_scores'] = df_test['cart_candidate_scores'].astype(object)
        df_test['order_candidate_scores'] = np.nan
        df_test['order_candidate_scores'] = df_test['order_candidate_scores'].astype(object)

        for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):

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

            df_test.at[idx, 'click_candidates'] = sorted_click_aids
            df_test.at[idx, 'cart_candidates'] = sorted_cart_aids
            df_test.at[idx, 'order_candidates'] = sorted_order_aids

            # Sort aids by their weights in descending order
            sorted_click_aid_weights = [weight for aid, weight in session_aid_click_weights.most_common(len(session_unique_aids))]
            sorted_cart_aid_weights = [weight for aid, weight in session_aid_cart_weights.most_common(len(session_unique_aids))]
            sorted_order_aid_weights = [weight for aid, weight in session_aid_order_weights.most_common(len(session_unique_aids))]

            df_test.at[idx, 'click_candidate_scores'] = sorted_click_aid_weights
            df_test.at[idx, 'cart_candidate_scores'] = sorted_cart_aid_weights
            df_test.at[idx, 'order_candidate_scores'] = sorted_order_aid_weights

        for event_type in ['click', 'cart', 'order']:

            candidate_columns = [f'{event_type}_candidates', f'{event_type}_candidate_scores']
            df_test_event = df_test.explode(candidate_columns)[['session'] + candidate_columns].rename(columns={
                f'{event_type}_candidates': 'candidates',
                f'{event_type}_candidate_scores': 'candidate_scores'
            })
            logging.info(
                f'''
                {event_type} recency weighted candidate generation
                {df_test_event.shape[0]} candidates are generated for {df_test_event["session"].nunique()} sessions
                Candidate scores - All {df_test_event["candidate_scores"].mean():.4f}
                '''
            )
            df_test_event['candidates'] = df_test_event['candidates'].astype(np.uint64)
            df_test_event['candidate_scores'] = df_test_event['candidate_scores'].astype(np.float32)
            df_test_event.to_pickle(candidate_directory / f'{event_type}_recency_weighted_test.pkl')

    else:
        raise ValueError('Invalid mode')
