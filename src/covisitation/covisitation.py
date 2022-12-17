import sys
import logging
import argparse
import pathlib
import json
from collections import defaultdict, Counter
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
import settings
import metrics


def covisitation_df_to_dict(df):

    """
    Convert covisitation dataframe to dictionary

    Parameters
    ----------
    df: pandas.DataFrame
        Covisitation dataframe

    Returns
    -------
    covisitation: dict
        Covisitation dictionary
    """

    return df.groupby('aid_x').aid_y.apply(list).to_dict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    frequency_directory = pathlib.Path(settings.DATA / 'aid_frequencies')

    with open(settings.DATA / frequency_directory / 'all_20_most_frequent_click_aids.json') as f:
        all_20_most_frequent_click_aids = list(json.load(f).keys())

    with open(settings.DATA / frequency_directory / 'all_20_most_frequent_cart_aids.json') as f:
        all_20_most_frequent_cart_aids = list(json.load(f).keys())

    with open(settings.DATA / frequency_directory / 'all_20_most_frequent_order_aids.json') as f:
        all_20_most_frequent_order_aids = list(json.load(f).keys())

    logging.info(f'Loaded top 20 most frequent aids from entire dataset for clicks, orders and carts')

    covisitation_directory = pathlib.Path(settings.DATA / 'covisitation')

    top_20_clicks = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'top_20_test_clicks_v1_0.pqt')))
    for i in range(1, 4):
        top_20_clicks.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / f'top_20_test_clicks_v1_{i}.pqt'))))

    top_20_buys = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / f'top_15_test_carts_orders_v1_0.pqt')))
    for i in range(1, 4):
        top_20_buys.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / f'top_15_test_carts_orders_v1_{i}.pqt'))))

    top_20_buy2buy = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / f'top_15_test_buy2buy_v1_0.pqt')))

    logging.info(f'Loaded top 20 covisitation statistics from entire dataset for clicks, orders and carts')

    if args.mode == 'validation':

        logging.info('Running covisitation model in validation mode')
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
            session_unique_aids = list(np.unique(session_aids))
            session_unique_cart_and_order_aids = np.unique(np.array(session_aids)[np.array(session_event_types) >= 1]).tolist()

            if len(session_unique_aids) >= 20:

                # Calculate click, cart and order weights based on recency
                click_weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
                cart_and_order_weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
                session_aid_click_weights = defaultdict(lambda: 0)
                session_aid_cart_and_order_weights = defaultdict(lambda: 0)

                for aid, event_type, click_weight, cart_and_order_weight in zip(session_aids, session_event_types, click_weights, cart_and_order_weights):
                    session_aid_click_weights[aid] += (click_weight * event_type_coefficient[event_type])
                    session_aid_cart_and_order_weights[aid] += (cart_and_order_weight * event_type_coefficient[event_type])

                # Sort click aids by their weights in descending order and take top 20 aids
                sorted_click_aids = [aid for aid, weight in sorted(session_aid_click_weights.items(), key=lambda item: -item[1])][:20]

                # Concatenate all covisited cart and order aids of session cart and order aids
                covisited_carts_and_orders = list(itertools.chain(*[top_20_buy2buy[aid] for aid in session_unique_cart_and_order_aids if aid in top_20_buy2buy]))
                # Increase weights of covisited cart and order aids
                for aid in covisited_carts_and_orders:
                    session_aid_cart_and_order_weights[aid] += 0.1

                # Sort cart and order aids by their weights in descending order and take top 20 aids
                sorted_cart_and_order_aids = [aid for aid, weight in sorted(session_aid_cart_and_order_weights.items(), key=lambda item: -item[1])][:20]

                click_predictions = sorted_click_aids
                cart_predictions = sorted_cart_and_order_aids
                order_predictions = sorted_cart_and_order_aids

                df_train_labels.at[idx, 'click_predictions'] = click_predictions
                df_train_labels.at[idx, 'cart_predictions'] = cart_predictions
                df_train_labels.at[idx, 'order_predictions'] = order_predictions

            else:

                # Concatenate all covisited click aids of session aids
                covisited_clicks = list(itertools.chain(*[top_20_clicks[aid] for aid in session_unique_aids if aid in top_20_clicks]))
                # Select top 20 covisited aids that are not in session aids
                covisited_clicks = [aid for aid, count in Counter(covisited_clicks).most_common(20) if aid not in session_unique_aids]
                click_predictions = session_unique_aids + covisited_clicks + all_20_most_frequent_click_aids

                covisited_carts_and_orders = []
                covisited_carts_and_orders += list(itertools.chain(*[top_20_buys[aid] for aid in session_unique_aids if aid in top_20_buys]))
                covisited_carts_and_orders += list(itertools.chain(*[top_20_buy2buy[aid] for aid in session_unique_cart_and_order_aids if aid in top_20_buy2buy]))
                covisited_carts_and_orders = [aid for aid, count in Counter(covisited_carts_and_orders).most_common(20) if aid not in session_unique_aids]
                cart_and_order_predictions = session_unique_aids + covisited_carts_and_orders + all_20_most_frequent_order_aids

                df_train_labels.at[idx, 'click_predictions'] = click_predictions[:20]
                df_train_labels.at[idx, 'cart_predictions'] = cart_and_order_predictions[:20]
                df_train_labels.at[idx, 'order_predictions'] = cart_and_order_predictions[:20]

            df_train_labels.at[idx, 'click_recall'] = metrics.click_recall(row['click_labels'], df_train_labels.at[idx, 'click_predictions'])
            df_train_labels.at[idx, 'cart_recall'] = metrics.cart_order_recall(row['cart_labels'], df_train_labels.at[idx, 'cart_predictions'])
            df_train_labels.at[idx, 'order_recall'] = metrics.cart_order_recall(row['order_labels'], df_train_labels.at[idx, 'order_predictions'])

        df_train_labels['recall'] = (df_train_labels['click_recall'] * 0.1) + (df_train_labels['cart_recall'] * 0.3) + (df_train_labels['order_recall'] * 0.6)
        mean_click_recall = df_train_labels['click_recall'].mean()
        mean_cart_recall = df_train_labels['cart_recall'].mean()
        mean_order_recall = df_train_labels['order_recall'].mean()
        mean_weighted_recall = df_train_labels['recall'].mean()

        logging.info(
            f'''
            covisitation model validation scores
            clicks - n: {df_train_labels["click_recall"].notna().sum()} recall@20: {mean_click_recall:.4f}
            carts - n: {df_train_labels["cart_recall"].notna().sum()} recall@20: {mean_cart_recall:.4f}
            orders - n: {df_train_labels["order_recall"].notna().sum()} recall@20: {mean_order_recall:.4f}
            weighted recall@20: {mean_weighted_recall:.4f}
            '''
        )

    elif args.mode == 'submission':

        logging.info('Running covisitation model in submission mode')
        df_test = pd.read_pickle(settings.DATA / 'test.pkl')
        logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

        # Create a directory for saving submission file
        submissions_directory = pathlib.Path(settings.DATA / 'submissions')
        submissions_directory.mkdir(parents=True, exist_ok=True)

        event_type_coefficient = {0: 1, 1: 3, 2: 6}
        df_test = df_test.groupby('session')[['aid', 'type']].agg(list).reset_index()
        test_predictions = []

        for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):

            session_aids = row['aid']
            session_event_types = row['type']
            session_unique_aids = list(np.unique(session_aids))
            session_unique_cart_and_order_aids = np.unique(np.array(session_aids)[np.array(session_event_types) >= 1]).tolist()

            if len(session_unique_aids) >= 20:

                # Calculate click, cart and order weights based on recency
                click_weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
                cart_and_order_weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
                session_aid_click_weights = defaultdict(lambda: 0)
                session_aid_cart_and_order_weights = defaultdict(lambda: 0)

                for aid, event_type, click_weight, cart_and_order_weight in zip(session_aids, session_event_types, click_weights, cart_and_order_weights):
                    session_aid_click_weights[aid] += click_weight * event_type_coefficient[event_type]
                    session_aid_cart_and_order_weights[aid] += cart_and_order_weight * event_type_coefficient[event_type]

                # Sort click aids by their weights in descending order and take top 20 aids
                click_predictions = [aid for aid, weight in sorted(session_aid_click_weights.items(), key=lambda item: -item[1])][:20]

                # Concatenate all covisited cart and order aids of session cart and order aids
                covisited_carts_and_orders = list(itertools.chain(*[top_20_buy2buy[aid] for aid in session_unique_cart_and_order_aids if aid in top_20_buy2buy]))
                # Increase weights of covisited cart and order aids
                for aid in covisited_carts_and_orders:
                    session_aid_cart_and_order_weights[aid] += 0.1

                # Sort cart and order aids by their weights in descending order and take top 20 aids
                cart_and_order_predictions = [aid for aid, weight in sorted(session_aid_cart_and_order_weights.items(), key=lambda item: -item[1])][:20]

            else:

                # Concatenate all covisited click aids of session aids
                covisited_clicks = list(itertools.chain(*[top_20_clicks[aid] for aid in session_unique_aids if aid in top_20_clicks]))
                # Select top 20 covisited aids that are not in session aids
                covisited_clicks = [aid for aid, count in Counter(covisited_clicks).most_common(20) if aid not in session_unique_aids]
                click_predictions = session_unique_aids + covisited_clicks + all_20_most_frequent_click_aids

                covisited_carts_and_orders = []
                covisited_carts_and_orders += list(itertools.chain(*[top_20_buys[aid] for aid in session_unique_aids if aid in top_20_buys]))
                covisited_carts_and_orders += list(itertools.chain(*[top_20_buy2buy[aid] for aid in session_unique_cart_and_order_aids if aid in top_20_buy2buy]))
                covisited_carts_and_orders = [aid for aid, count in Counter(covisited_carts_and_orders).most_common(20) if aid not in session_unique_aids]
                cart_and_order_predictions = session_unique_aids + covisited_carts_and_orders + all_20_most_frequent_order_aids

            for event_type, predictions in zip(['click', 'cart', 'order'], [click_predictions, cart_and_order_predictions, cart_and_order_predictions]):
                test_predictions.append({
                    'session_type': f'{row["session"]}_{event_type}s',
                    'labels': ' '.join([str(aid) for aid in predictions[:20]])
                })

        df_test_predictions = pd.DataFrame(test_predictions)
        df_test_predictions.to_csv(submissions_directory / 'covisitation_submission.csv.gz', index=False, compression='gzip')

    else:
        raise ValueError('Invalid mode')