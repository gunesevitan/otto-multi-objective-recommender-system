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

    return df.groupby('aid_x')['aid_y'].apply(list).to_dict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    frequency_directory = pathlib.Path(settings.DATA / 'aid_frequencies')

    with open(settings.DATA / frequency_directory / 'all_20_most_frequent_click_aids.json') as f:
        most_frequent_click_aids = list(json.load(f).keys())

    with open(settings.DATA / frequency_directory / 'all_20_most_frequent_cart_aids.json') as f:
        most_frequent_cart_aids = list(json.load(f).keys())

    with open(settings.DATA / frequency_directory / 'all_20_most_frequent_order_aids.json') as f:
        most_frequent_order_aids = list(json.load(f).keys())

    logging.info(f'Loaded top 20 most frequent aids from entire dataset for clicks, orders and carts')

    covisitation_directory = pathlib.Path(settings.DATA / 'covisitation')

    top_clicks = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'top_15_clicks_0.pqt')))
    for i in range(1, 4):
        top_clicks.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / f'top_15_clicks_{i}.pqt'))))

    top_buys = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'top_20_carts_orders_0.pqt')))
    for i in range(1, 4):
        top_buys.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / f'top_20_carts_orders_{i}.pqt'))))

    top_buy2buy = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'top_20_buy2buy_0.pqt')))

    logging.info(f'Loaded top covisitation statistics from entire dataset for clicks, orders and carts')
    event_type_coefficient = {0: 0.5, 1: 9, 2: 0.5}

    if args.mode == 'validation':

        logging.info('Running covisitation model in validation mode')
        df_train_labels = pd.read_pickle(settings.DATA / 'train_labels.pkl')
        logging.info(f'Train Labels Shape: {df_train_labels.shape} - Memory Usage: {df_train_labels.memory_usage().sum() / 1024 ** 2:.2f} MB')

        # Cut session aids and event types from their cutoff index
        df_train_labels['aid'] = df_train_labels[['aid', 'session_cutoff_idx']].apply(lambda x: x['aid'][:x['session_cutoff_idx'] + 1], axis=1)
        df_train_labels['type'] = df_train_labels[['type', 'session_cutoff_idx']].apply(lambda x: x['type'][:x['session_cutoff_idx'] + 1], axis=1)

        # Specify prediction types for different models
        df_train_labels['session_unique_aid_count'] = df_train_labels['aid'].apply(lambda session_aids: len(set(session_aids)))
        recency_weight_predictions_idx = df_train_labels['session_unique_aid_count'] >= 20
        df_train_labels.loc[recency_weight_predictions_idx, 'prediction_type'] = 'recency_weight'
        df_train_labels.loc[~recency_weight_predictions_idx, 'prediction_type'] = 'covisitation'
        del recency_weight_predictions_idx
        logging.info(f'Prediction type distribution: {json.dumps(df_train_labels["prediction_type"].value_counts().to_dict(), indent=2)}')

        df_train_labels['click_predictions'] = np.nan
        df_train_labels['click_predictions'] = df_train_labels['click_predictions'].astype(object)
        df_train_labels['cart_predictions'] = np.nan
        df_train_labels['cart_predictions'] = df_train_labels['cart_predictions'].astype(object)
        df_train_labels['order_predictions'] = np.nan
        df_train_labels['order_predictions'] = df_train_labels['order_predictions'].astype(object)

        recency_weight_predictions_idx = df_train_labels['prediction_type'] == 'recency_weight'
        for idx, row in tqdm(df_train_labels.loc[recency_weight_predictions_idx].iterrows(), total=df_train_labels.loc[recency_weight_predictions_idx].shape[0]):

            session_aids = row['aid']
            session_event_types = row['type']
            session_unique_aids = list(dict.fromkeys(session_aids[::-1]))
            session_unique_click_and_cart_aids = np.unique(np.array(session_aids)[np.array(session_event_types) <= 1]).tolist()
            session_unique_cart_and_order_aids = np.unique(np.array(session_aids)[np.array(session_event_types) >= 1]).tolist()

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

            # Sort click aids by their weights in descending order
            sorted_click_aids = [aid for aid, weight in session_aid_click_weights.most_common(20)]

            # Concatenate all covisited click and cart aids and increase cart weights based on covisitation
            covisited_click_and_cart_aids = list(itertools.chain(*[top_buys[aid] for aid in session_unique_click_and_cart_aids if aid in top_buys]))
            for aid in covisited_click_and_cart_aids:
                session_aid_cart_weights[aid] += 0.25

            # Sort cart aids by their weights in descending order
            sorted_cart_aids = [aid for aid, weight in session_aid_cart_weights.most_common(20)]

            # Concatenate all covisited cart and order aids and increase order weights based on covisitation
            covisited_cart_and_order_aids = list(itertools.chain(*[top_buy2buy[aid] for aid in session_unique_cart_and_order_aids if aid in top_buy2buy]))
            for aid in covisited_cart_and_order_aids:
                session_aid_order_weights[aid] += 0.50

            # Sort order aids by their weights in descending order
            sorted_order_aids = [aid for aid, weight in session_aid_order_weights.most_common(20)]

            click_predictions = sorted_click_aids
            cart_predictions = sorted_cart_aids
            order_predictions = sorted_order_aids

            df_train_labels.at[idx, 'click_predictions'] = click_predictions
            df_train_labels.at[idx, 'cart_predictions'] = sorted_cart_aids
            df_train_labels.at[idx, 'order_predictions'] = sorted_order_aids

        logging.info(f'{recency_weight_predictions_idx.sum()} sessions are predicted with recency weight')

        covisitation_predictions_idx = ~recency_weight_predictions_idx
        for idx, row in tqdm(df_train_labels.loc[covisitation_predictions_idx].iterrows(), total=df_train_labels.loc[covisitation_predictions_idx].shape[0]):

            session_aids = row['aid']
            session_event_types = row['type']
            session_unique_aids = list(dict.fromkeys(session_aids[::-1]))
            session_unique_click_and_cart_aids = np.unique(np.array(session_aids)[np.array(session_event_types) <= 1]).tolist()
            session_unique_cart_and_order_aids = np.unique(np.array(session_aids)[np.array(session_event_types) >= 1]).tolist()

            # Concatenate all covisited click aids and select most common ones
            covisited_click_aids = list(itertools.chain(*[top_clicks[aid] for aid in session_unique_aids if aid in top_clicks]))
            sorted_click_aids = [aid for aid, count in Counter(covisited_click_aids).most_common(20) if aid not in session_unique_aids]

            covisited_click_and_cart_aids = []
            covisited_click_and_cart_aids += covisited_click_aids
            covisited_click_and_cart_aids += list(itertools.chain(*[top_buys[aid] for aid in session_unique_aids if aid in top_buys]))
            sorted_cart_aids = [aid for aid, count in Counter(covisited_click_and_cart_aids).most_common(20) if aid not in session_unique_aids]

            covisited_cart_and_order_aids = []
            covisited_cart_and_order_aids += list(itertools.chain(*[top_buys[aid] for aid in session_unique_aids if aid in top_buys]))
            covisited_cart_and_order_aids += list(itertools.chain(*[top_buy2buy[aid] for aid in session_unique_cart_and_order_aids if aid in top_buy2buy]))
            sorted_order_aids = [aid for aid, count in Counter(covisited_cart_and_order_aids).most_common(20) if aid not in session_unique_aids]

            click_predictions = session_unique_aids + sorted_click_aids
            click_predictions = click_predictions + most_frequent_click_aids[:20 - len(click_predictions)]
            cart_predictions = session_unique_aids + sorted_cart_aids
            cart_predictions = cart_predictions + most_frequent_cart_aids[:20 - len(cart_predictions)]
            order_predictions = session_unique_aids + sorted_order_aids
            order_predictions = order_predictions + most_frequent_cart_aids[:20 - len(order_predictions)]

            df_train_labels.at[idx, 'click_predictions'] = click_predictions
            df_train_labels.at[idx, 'cart_predictions'] = cart_predictions
            df_train_labels.at[idx, 'order_predictions'] = order_predictions

        logging.info(f'{covisitation_predictions_idx.sum()} sessions are predicted with covisitation')

        df_train_labels['click_recall'] = df_train_labels[['click_labels', 'click_predictions']].apply(lambda x: metrics.click_recall(x['click_labels'], x['click_predictions']), axis=1)
        df_train_labels['cart_recall'] = df_train_labels[['cart_labels', 'cart_predictions']].apply(lambda x: metrics.cart_order_recall(x['cart_labels'], x['cart_predictions']), axis=1)
        df_train_labels['order_recall'] = df_train_labels[['order_labels', 'order_predictions']].apply(lambda x: metrics.cart_order_recall(x['order_labels'], x['order_predictions']), axis=1)
        df_train_labels['weighted_recall'] = (df_train_labels['click_recall'] * 0.1) + (df_train_labels['cart_recall'] * 0.3) + (df_train_labels['order_recall'] * 0.6)

        mean_click_recall = df_train_labels['click_recall'].mean()
        mean_cart_recall = df_train_labels['cart_recall'].mean()
        mean_order_recall = df_train_labels['order_recall'].mean()
        mean_weighted_recall = df_train_labels['weighted_recall'].mean()

        logging.info(
            f'''
            Covisitation model validation scores
            clicks - n: {df_train_labels["click_recall"].notna().sum()} recall@20: {mean_click_recall:.4f}
            carts - n: {df_train_labels["cart_recall"].notna().sum()} recall@20: {mean_cart_recall:.4f}
            orders - n: {df_train_labels["order_recall"].notna().sum()} recall@20: {mean_order_recall:.4f}
            weighted recall@20: {mean_weighted_recall:.4f}
            '''
        )

    elif args.mode == 'submission':

        test_predictions = []

        # Create a directory for saving submission file
        submissions_directory = pathlib.Path(settings.DATA / 'submissions')
        submissions_directory.mkdir(parents=True, exist_ok=True)

        logging.info('Running covisitation model in submission mode')
        df_test = pd.read_pickle(settings.DATA / 'test.pkl')
        df_test = df_test.groupby('session')[['aid', 'type']].agg(list).reset_index()
        logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

        # Specify prediction types for different models
        df_test['session_unique_aid_count'] = df_test['aid'].apply(lambda session_aids: len(set(session_aids)))
        recency_weight_predictions_idx = df_test['session_unique_aid_count'] >= 20
        df_test.loc[recency_weight_predictions_idx, 'prediction_type'] = 'recency_weight'
        df_test.loc[~recency_weight_predictions_idx, 'prediction_type'] = 'covisitation'
        del recency_weight_predictions_idx
        logging.info(f'Prediction type distribution: {json.dumps(df_test["prediction_type"].value_counts().to_dict(), indent=2)}')

        event_type_coefficient = {0: 0.5, 1: 9, 2: 0.5}
        recency_weight_predictions_idx = df_test['prediction_type'] == 'recency_weight'
        for idx, row in tqdm(df_test.loc[recency_weight_predictions_idx].iterrows(), total=df_test.loc[recency_weight_predictions_idx].shape[0]):

            session_aids = row['aid']
            session_event_types = row['type']
            session_unique_aids = list(dict.fromkeys(session_aids[::-1]))
            session_unique_click_and_cart_aids = np.unique(np.array(session_aids)[np.array(session_event_types) <= 1]).tolist()
            session_unique_cart_and_order_aids = np.unique(np.array(session_aids)[np.array(session_event_types) >= 1]).tolist()

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

            # Sort click aids by their weights in descending order
            sorted_click_aids = [aid for aid, weight in session_aid_click_weights.most_common(20)]

            # Concatenate all covisited click and cart aids and increase cart weights based on covisitation
            covisited_click_and_cart_aids = list(itertools.chain(*[top_buys[aid] for aid in session_unique_click_and_cart_aids if aid in top_buys]))
            for aid in covisited_click_and_cart_aids:
                session_aid_cart_weights[aid] += 0.1

            # Sort cart aids by their weights in descending order
            sorted_cart_aids = [aid for aid, weight in session_aid_cart_weights.most_common(20)]

            # Concatenate all covisited cart and order aids and increase order weights based on covisitation
            covisited_cart_and_order_aids = list(itertools.chain(*[top_buy2buy[aid] for aid in session_unique_cart_and_order_aids if aid in top_buy2buy]))
            for aid in covisited_cart_and_order_aids:
                session_aid_order_weights[aid] += 0.1

            # Sort order aids by their weights in descending order
            sorted_order_aids = [aid for aid, weight in session_aid_order_weights.most_common(20)]

            click_predictions = sorted_click_aids
            cart_predictions = sorted_cart_aids
            order_predictions = sorted_order_aids

            for event_type, predictions in zip(['click', 'cart', 'order'], [click_predictions, cart_predictions, order_predictions]):
                test_predictions.append({
                    'session_type': f'{row["session"]}_{event_type}s',
                    'labels': ' '.join([str(aid) for aid in predictions])
                })

        logging.info(f'{recency_weight_predictions_idx.sum()} sessions are predicted with recency weight')

        covisitation_predictions_idx = ~recency_weight_predictions_idx
        for idx, row in tqdm(df_test.loc[covisitation_predictions_idx].iterrows(), total=df_test.loc[covisitation_predictions_idx].shape[0]):

            session_aids = row['aid']
            session_event_types = row['type']
            session_unique_aids = list(dict.fromkeys(session_aids[::-1]))
            session_unique_click_and_cart_aids = np.unique(np.array(session_aids)[np.array(session_event_types) <= 1]).tolist()
            session_unique_cart_and_order_aids = np.unique(np.array(session_aids)[np.array(session_event_types) >= 1]).tolist()

            # Concatenate all covisited click aids and select most common ones
            covisited_click_aids = list(itertools.chain(*[top_clicks[aid] for aid in session_unique_aids if aid in top_clicks]))
            sorted_click_aids = [aid for aid, count in Counter(covisited_click_aids).most_common(20) if aid not in session_unique_aids]

            covisited_click_and_cart_aids = []
            covisited_click_and_cart_aids += covisited_click_aids
            covisited_click_and_cart_aids += list(itertools.chain(*[top_buys[aid] for aid in session_unique_aids if aid in top_buys]))
            sorted_cart_aids = [aid for aid, count in Counter(covisited_click_and_cart_aids).most_common(20) if aid not in session_unique_aids]

            covisited_cart_and_order_aids = []
            covisited_cart_and_order_aids += list(itertools.chain(*[top_buys[aid] for aid in session_unique_aids if aid in top_buys]))
            covisited_cart_and_order_aids += list(itertools.chain(*[top_buy2buy[aid] for aid in session_unique_cart_and_order_aids if aid in top_buy2buy]))
            sorted_order_aids = [aid for aid, count in Counter(covisited_cart_and_order_aids).most_common(20) if aid not in session_unique_aids]

            click_predictions = session_unique_aids + sorted_click_aids
            click_predictions = click_predictions + most_frequent_click_aids[:20 - len(click_predictions)]
            cart_predictions = session_unique_aids + sorted_cart_aids
            cart_predictions = cart_predictions + most_frequent_cart_aids[:20 - len(cart_predictions)]
            order_predictions = session_unique_aids + sorted_order_aids
            order_predictions = order_predictions + most_frequent_order_aids[:20 - len(order_predictions)]

            for event_type, predictions in zip(['click', 'cart', 'order'], [click_predictions, cart_predictions, order_predictions]):
                test_predictions.append({
                    'session_type': f'{row["session"]}_{event_type}s',
                    'labels': ' '.join([str(aid) for aid in predictions[:20]])
                })

        logging.info(f'{covisitation_predictions_idx.sum()} sessions are predicted with covisitation')

        df_test_predictions = pd.DataFrame(test_predictions)
        logging.info(f'Prediction lengths {json.dumps(df_test_predictions["labels"].apply(lambda x: len(x.split())).value_counts().to_dict(), indent=2)}')
        df_test_predictions.to_csv(submissions_directory / 'covisitation_submission.csv.gz', index=False, compression='gzip')

    else:
        raise ValueError('Invalid mode')
