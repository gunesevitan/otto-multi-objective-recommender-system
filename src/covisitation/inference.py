import sys
import logging
import argparse
import pathlib
import json
from collections import Counter
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
import polars as pl

sys.path.append('..')
import settings


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
    covisitation_directory = pathlib.Path(settings.DATA / 'covisitation')

    event_type_coefficient = {0: 1, 1: 9, 2: 6}

    if args.mode == 'validation':

        with open(settings.DATA / frequency_directory / 'train_20_most_frequent_click_aids.json') as f:
            most_frequent_click_aids = [int(aid) for aid in list(json.load(f).keys())]

        with open(settings.DATA / frequency_directory / 'train_20_most_frequent_cart_aids.json') as f:
            most_frequent_cart_aids = [int(aid) for aid in list(json.load(f).keys())]

        with open(settings.DATA / frequency_directory / 'train_20_most_frequent_order_aids.json') as f:
            most_frequent_order_aids = [int(aid) for aid in list(json.load(f).keys())]

        logging.info(f'Loaded top 20 most frequent aids from entire dataset for clicks, orders and carts')

        top_time_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_15_time_weighted_0.pqt')))
        for i in range(1, 4):
            top_time_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_15_time_weighted_{i}.pqt'))))

        top_click_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_15_click_weighted_0.pqt')))
        for i in range(1, 4):
            top_click_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_15_click_weighted_{i}.pqt'))))

        top_cart_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_15_cart_weighted_0.pqt')))
        for i in range(1, 4):
            top_cart_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_15_cart_weighted_{i}.pqt'))))

        top_order_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_15_order_weighted_0.pqt')))
        for i in range(1, 4):
            top_order_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_15_order_weighted_{i}.pqt'))))

        top_click_cart_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_15_click_cart_0.pqt')))
        for i in range(1, 4):
            top_click_cart_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_15_click_cart_{i}.pqt'))))

        top_click_order_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_15_click_order_0.pqt')))
        for i in range(1, 4):
            top_click_order_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_15_click_order_{i}.pqt'))))

        top_cart_order_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_15_cart_order_0.pqt')))

        logging.info(f'Loaded top covisitation statistics from entire dataset for clicks, orders and carts')
        logging.info('Running covisitation model in validation mode')

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

        # Specify prediction types for different models
        df_val['session_unique_aid_count'] = df_val['aid'].apply(lambda session_aids: len(set(session_aids)))
        recency_weight_predictions_idx = df_val['session_unique_aid_count'] >= 20
        df_val.loc[recency_weight_predictions_idx, 'prediction_type'] = 'recency_weight'
        df_val.loc[~recency_weight_predictions_idx, 'prediction_type'] = 'covisitation'
        del recency_weight_predictions_idx
        logging.info(f'Prediction type distribution: {json.dumps(df_val["prediction_type"].value_counts().to_dict(), indent=2)}')

        df_val['click_predictions'] = np.nan
        df_val['click_predictions'] = df_val['click_predictions'].astype(object)
        df_val['cart_predictions'] = np.nan
        df_val['cart_predictions'] = df_val['cart_predictions'].astype(object)
        df_val['order_predictions'] = np.nan
        df_val['order_predictions'] = df_val['order_predictions'].astype(object)

        recency_weight_predictions_idx = df_val['prediction_type'] == 'recency_weight'
        for t in tqdm(df_val.loc[recency_weight_predictions_idx].itertuples(), total=df_val.loc[recency_weight_predictions_idx].shape[0]):

            session_aids = t.aid
            session_event_types = t.type
            session_unique_aids = list(dict.fromkeys(session_aids[::-1]))
            session_unique_click_aids = np.unique(np.array(session_aids)[np.array(session_event_types) == 0]).tolist()
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

            # Concatenate all covisited click aids and increase click weights based on covisitation
            covisited_clicks_aids = list(itertools.chain(*[top_time_weighted_covisitation[aid] for aid in session_unique_click_aids if aid in top_time_weighted_covisitation]))
            for aid in covisited_clicks_aids:
                session_aid_click_weights[aid] += 0.05

            # Sort click aids by their weights in descending order
            sorted_click_aids = [aid for aid, weight in session_aid_click_weights.most_common(20)]

            # Concatenate all covisited click and cart aids and increase cart weights based on covisitation
            cart_weighted_covisited_aids = list(itertools.chain(*[top_cart_weighted_covisitation[aid] for aid in session_unique_click_and_cart_aids if aid in top_cart_weighted_covisitation]))
            for aid in cart_weighted_covisited_aids:
                session_aid_cart_weights[aid] += 0.05

            # Sort cart aids by their weights in descending order
            sorted_cart_aids = [aid for aid, weight in session_aid_cart_weights.most_common(20)]

            # Concatenate all covisited cart and order aids and increase order weights based on covisitation
            covisited_cart_and_order_aids = list(itertools.chain(*[top_cart_order_covisitation[aid] for aid in session_unique_cart_and_order_aids if aid in top_cart_order_covisitation]))
            for aid in covisited_cart_and_order_aids:
                session_aid_order_weights[aid] += 0.15

            # Sort order aids by their weights in descending order
            sorted_order_aids = [aid for aid, weight in session_aid_order_weights.most_common(20)]

            df_val.at[t.Index, 'click_predictions'] = sorted_click_aids
            df_val.at[t.Index, 'cart_predictions'] = sorted_cart_aids
            df_val.at[t.Index, 'order_predictions'] = sorted_order_aids

        logging.info(f'{recency_weight_predictions_idx.sum()} sessions are predicted with recency weight')

        covisitation_predictions_idx = ~recency_weight_predictions_idx
        for t in tqdm(df_val.loc[covisitation_predictions_idx].itertuples(), total=df_val.loc[covisitation_predictions_idx].shape[0]):

            session_aids = t.aid
            session_event_types = t.type
            session_unique_aids = list(dict.fromkeys(session_aids[::-1]))
            session_unique_click_aids = np.unique(np.array(session_aids)[np.array(session_event_types) == 0]).tolist()
            session_unique_cart_aids = np.unique(np.array(session_aids)[np.array(session_event_types) == 1]).tolist()
            session_unique_order_aids = np.unique(np.array(session_aids)[np.array(session_event_types) == 2]).tolist()
            session_unique_click_and_cart_aids = np.unique(np.array(session_aids)[np.array(session_event_types) <= 1]).tolist()
            session_unique_cart_and_order_aids = np.unique(np.array(session_aids)[np.array(session_event_types) >= 1]).tolist()

            time_weighted_covisited_aids = list(itertools.chain(*[top_time_weighted_covisitation[aid] for aid in session_unique_aids if aid in top_time_weighted_covisitation]))
            click_weighted_covisited_aids = list(itertools.chain(*[top_click_weighted_covisitation[aid] for aid in session_unique_click_and_cart_aids if aid in top_click_weighted_covisitation]))
            cart_weighted_covisited_aids = list(itertools.chain(*[top_cart_weighted_covisitation[aid] for aid in session_unique_click_and_cart_aids if aid in top_cart_weighted_covisitation]))
            order_weighted_covisited_aids = list(itertools.chain(*[top_order_weighted_covisitation[aid] for aid in session_unique_cart_and_order_aids if aid in top_order_weighted_covisitation]))
            click_cart_covisited_aids = list(itertools.chain(*[top_click_cart_covisitation[aid] for aid in session_unique_click_and_cart_aids if aid in top_click_cart_covisitation]))
            cart_order_covisited_aids = list(itertools.chain(*[top_cart_order_covisitation[aid] for aid in session_unique_click_and_cart_aids if aid in top_cart_order_covisitation]))

            # Concatenate all covisited click aids and select most common ones
            covisited_click_aids = time_weighted_covisited_aids + click_weighted_covisited_aids + cart_weighted_covisited_aids + click_cart_covisited_aids + cart_order_covisited_aids
            sorted_click_aids = [aid for aid, count in Counter(covisited_click_aids).most_common(20) if aid not in session_unique_aids]

            # Concatenate all covisited cart aids and select most common ones
            covisited_cart_aids = time_weighted_covisited_aids + cart_weighted_covisited_aids + cart_order_covisited_aids
            sorted_cart_aids = [aid for aid, count in Counter(covisited_cart_aids).most_common(20) if aid not in session_unique_aids]

            covisited_order_aids = time_weighted_covisited_aids + cart_weighted_covisited_aids + cart_order_covisited_aids
            sorted_order_aids = [aid for aid, count in Counter(covisited_order_aids).most_common(20) if aid not in session_unique_aids]

            click_predictions = session_unique_aids + sorted_click_aids[:20 - len(session_unique_aids)]
            click_predictions = click_predictions + most_frequent_click_aids[:20 - len(click_predictions)]
            cart_predictions = session_unique_aids + sorted_cart_aids[:20 - len(session_unique_aids)]
            cart_predictions = cart_predictions + most_frequent_cart_aids[:20 - len(cart_predictions)]
            order_predictions = session_unique_aids + sorted_order_aids[:20 - len(session_unique_aids)]
            order_predictions = order_predictions + most_frequent_order_aids[:20 - len(order_predictions)]

            df_val.at[t.Index, 'click_predictions'] = click_predictions
            df_val.at[t.Index, 'cart_predictions'] = cart_predictions
            df_val.at[t.Index, 'order_predictions'] = order_predictions

        logging.info(f'{covisitation_predictions_idx.sum()} sessions are predicted with covisitation')

        df_val['click_hits'] = pl.DataFrame(df_val[['click_predictions', 'click_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        click_recall = df_val['click_hits'].sum() / df_val['click_labels'].apply(len).clip(0, 20).sum()
        df_val['cart_hits'] = pl.DataFrame(df_val[['cart_predictions', 'cart_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        cart_recall = df_val['cart_hits'].sum() / df_val['cart_labels'].apply(len).clip(0, 20).sum()
        df_val['order_hits'] = pl.DataFrame(df_val[['order_predictions', 'order_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        order_recall = df_val['order_hits'].sum() / df_val['order_labels'].apply(len).clip(0, 20).sum()
        weighted_recall = (click_recall * 0.1) + (cart_recall * 0.3) + (order_recall * 0.6)

        logging.info(
            f'''
            Covisitation model validation scores
            clicks - n: {df_val["click_labels"].apply(lambda x: len(x) > 0).sum()} recall@20: {click_recall:.6f}
            carts - n: {df_val["cart_labels"].apply(lambda x: len(x) > 0).sum()} recall@20: {cart_recall:.6f}
            orders - n: {df_val["order_labels"].apply(lambda x: len(x) > 0).sum()} recall@20: {order_recall:.6f}
            weighted recall@20: {weighted_recall:.6f}
            '''
        )

    elif args.mode == 'submission':

        with open(settings.DATA / frequency_directory / 'test_20_most_frequent_click_aids.json') as f:
            most_frequent_click_aids = [int(aid) for aid in list(json.load(f).keys())]

        with open(settings.DATA / frequency_directory / 'test_20_most_frequent_cart_aids.json') as f:
            most_frequent_cart_aids = [int(aid) for aid in list(json.load(f).keys())]

        with open(settings.DATA / frequency_directory / 'test_20_most_frequent_order_aids.json') as f:
            most_frequent_order_aids = [int(aid) for aid in list(json.load(f).keys())]

        logging.info(f'Loaded top 20 most frequent aids from entire dataset for clicks, orders and carts')

        top_time_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_15_time_weighted_0.pqt')))
        for i in range(1, 6):
            top_time_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_15_time_weighted_{i}.pqt'))))

        top_click_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_15_click_weighted_0.pqt')))
        for i in range(1, 6):
            top_click_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_15_click_weighted_{i}.pqt'))))

        top_cart_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_15_cart_weighted_0.pqt')))
        for i in range(1, 6):
            top_cart_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_15_cart_weighted_{i}.pqt'))))

        top_order_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_15_order_weighted_0.pqt')))
        for i in range(1, 6):
            top_order_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_15_order_weighted_{i}.pqt'))))

        top_click_cart_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_15_click_cart_0.pqt')))
        for i in range(1, 6):
            top_click_cart_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_15_click_cart_{i}.pqt'))))

        top_click_order_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_15_click_order_0.pqt')))
        for i in range(1, 6):
            top_click_order_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_15_click_order_{i}.pqt'))))

        top_cart_order_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_15_cart_order_0.pqt')))
        for i in range(1, 2):
            top_cart_order_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_15_cart_order_{i}.pqt'))))

        logging.info(f'Loaded top covisitation statistics from entire dataset for clicks, orders and carts')

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

        event_type_coefficient = {0: 1, 1: 9, 2: 6}
        recency_weight_predictions_idx = df_test['prediction_type'] == 'recency_weight'
        for t in tqdm(df_test.loc[recency_weight_predictions_idx].itertuples(), total=df_test.loc[recency_weight_predictions_idx].shape[0]):

            session_aids = t.aid
            session_event_types = t.type
            session_unique_aids = list(dict.fromkeys(session_aids[::-1]))
            session_unique_click_aids = np.unique(np.array(session_aids)[np.array(session_event_types) == 0]).tolist()
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

            # Concatenate all covisited click aids and increase click weights based on covisitation
            covisited_clicks_aids = list(itertools.chain(*[top_time_weighted_covisitation[aid] for aid in session_unique_click_aids if aid in top_time_weighted_covisitation]))
            for aid in covisited_clicks_aids:
                session_aid_click_weights[aid] += 0.05

            # Sort click aids by their weights in descending order
            sorted_click_aids = [aid for aid, weight in session_aid_click_weights.most_common(20)]

            # Concatenate all covisited click and cart aids and increase cart weights based on covisitation
            cart_weighted_covisited_aids = list(itertools.chain(*[top_cart_weighted_covisitation[aid] for aid in session_unique_click_and_cart_aids if aid in top_cart_weighted_covisitation]))
            for aid in cart_weighted_covisited_aids:
                session_aid_cart_weights[aid] += 0.05

            # Sort cart aids by their weights in descending order
            sorted_cart_aids = [aid for aid, weight in session_aid_cart_weights.most_common(20)]

            # Concatenate all covisited cart and order aids and increase order weights based on covisitation
            covisited_cart_and_order_aids = list(itertools.chain(*[top_cart_order_covisitation[aid] for aid in session_unique_cart_and_order_aids if aid in top_cart_order_covisitation]))
            for aid in covisited_cart_and_order_aids:
                session_aid_order_weights[aid] += 0.15

            # Sort order aids by their weights in descending order
            sorted_order_aids = [aid for aid, weight in session_aid_order_weights.most_common(20)]

            for event_type, predictions in zip(['click', 'cart', 'order'], [sorted_click_aids, sorted_cart_aids, sorted_order_aids]):
                test_predictions.append({
                    'session_type': f'{t.session}_{event_type}s',
                    'labels': ' '.join([str(aid) for aid in predictions])
                })

        logging.info(f'{recency_weight_predictions_idx.sum()} sessions are predicted with recency weight')

        covisitation_predictions_idx = ~recency_weight_predictions_idx
        for t in tqdm(df_test.loc[covisitation_predictions_idx].itertuples(), total=df_test.loc[covisitation_predictions_idx].shape[0]):

            session_aids = t.aid
            session_event_types = t.type
            session_unique_aids = list(dict.fromkeys(session_aids[::-1]))
            session_unique_click_aids = np.unique(np.array(session_aids)[np.array(session_event_types) == 0]).tolist()
            session_unique_cart_aids = np.unique(np.array(session_aids)[np.array(session_event_types) == 1]).tolist()
            session_unique_order_aids = np.unique(np.array(session_aids)[np.array(session_event_types) == 2]).tolist()
            session_unique_click_and_cart_aids = np.unique(np.array(session_aids)[np.array(session_event_types) <= 1]).tolist()
            session_unique_cart_and_order_aids = np.unique(np.array(session_aids)[np.array(session_event_types) >= 1]).tolist()

            time_weighted_covisited_aids = list(itertools.chain(*[top_time_weighted_covisitation[aid] for aid in session_unique_aids if aid in top_time_weighted_covisitation]))
            click_weighted_covisited_aids = list(itertools.chain(*[top_click_weighted_covisitation[aid] for aid in session_unique_click_and_cart_aids if aid in top_click_weighted_covisitation]))
            cart_weighted_covisited_aids = list(itertools.chain(*[top_cart_weighted_covisitation[aid] for aid in session_unique_click_and_cart_aids if aid in top_cart_weighted_covisitation]))
            order_weighted_covisited_aids = list(itertools.chain(*[top_order_weighted_covisitation[aid] for aid in session_unique_cart_and_order_aids if aid in top_order_weighted_covisitation]))
            click_cart_covisited_aids = list(itertools.chain(*[top_click_cart_covisitation[aid] for aid in session_unique_click_and_cart_aids if aid in top_click_cart_covisitation]))
            cart_order_covisited_aids = list(itertools.chain(*[top_cart_order_covisitation[aid] for aid in session_unique_click_and_cart_aids if aid in top_cart_order_covisitation]))

            # Concatenate all covisited click aids and select most common ones
            covisited_click_aids = time_weighted_covisited_aids + click_weighted_covisited_aids + cart_weighted_covisited_aids + click_cart_covisited_aids + cart_order_covisited_aids
            sorted_click_aids = [aid for aid, count in Counter(covisited_click_aids).most_common(20) if aid not in session_unique_aids]

            # Concatenate all covisited cart aids and select most common ones
            covisited_cart_aids = time_weighted_covisited_aids + cart_weighted_covisited_aids + cart_order_covisited_aids
            sorted_cart_aids = [aid for aid, count in Counter(covisited_cart_aids).most_common(20) if aid not in session_unique_aids]

            covisited_order_aids = time_weighted_covisited_aids + cart_weighted_covisited_aids + cart_order_covisited_aids
            sorted_order_aids = [aid for aid, count in Counter(covisited_order_aids).most_common(20) if aid not in session_unique_aids]

            click_predictions = session_unique_aids + sorted_click_aids[:20 - len(session_unique_aids)]
            click_predictions = click_predictions + most_frequent_click_aids[:20 - len(click_predictions)]
            cart_predictions = session_unique_aids + sorted_cart_aids[:20 - len(session_unique_aids)]
            cart_predictions = cart_predictions + most_frequent_cart_aids[:20 - len(cart_predictions)]
            order_predictions = session_unique_aids + sorted_order_aids[:20 - len(session_unique_aids)]
            order_predictions = order_predictions + most_frequent_order_aids[:20 - len(order_predictions)]

            for event_type, predictions in zip(['click', 'cart', 'order'], [click_predictions, cart_predictions, order_predictions]):
                test_predictions.append({
                    'session_type': f'{t.session}_{event_type}s',
                    'labels': ' '.join([str(aid) for aid in predictions[:20]])
                })

        logging.info(f'{covisitation_predictions_idx.sum()} sessions are predicted with covisitation')

        df_test_predictions = pd.DataFrame(test_predictions)
        logging.info(f'Prediction lengths {json.dumps(df_test_predictions["labels"].apply(lambda x: len(x.split())).value_counts().to_dict(), indent=2)}')
        df_test_predictions.to_csv(submissions_directory / 'covisitation_submission.csv.gz', index=False, compression='gzip')

    else:
        raise ValueError('Invalid mode')
