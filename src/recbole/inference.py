import json
import sys
import logging
import argparse
import torch
import yaml
import pathlib
import glob
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import fasttext
from annoy import AnnoyIndex
from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction
from recbole.model.general_recommender import *

sys.path.append('..')
import settings
import metrics


def predict(sessions, session_aids, topk, pad_length, dataset, model):

    """
    Predict aids using given recbole model

    Parameters
    ----------
    sessions: pd.Series of shape (batch_size)
        Series of sessions

    session_aids: np.ndarray of shape (n_aids)
        Array of aids

    topk: int
        Number of items to predict

    pad_length: int
        Number of padding tokens to add

    dataset: recbole.Dataset
        Constructed dataset

    model: recbole.Model
        Recbole model

    Returns
    -------
    prediction_aids: np.ndarray of shape (batch_size, topk)
        Top-k predicted aids

    prediction_scores: np.ndarray of shape (batch_size, topk)
        Top-k predicted scores
    """

    if isinstance(model, (BPR, LINE)):
        interaction = Interaction({
            'session': torch.tensor(dataset.token2id(dataset.uid_field, sessions.astype(str).values))
        })
    else:
        padded_item_sequence = torch.nn.functional.pad(
            torch.tensor(dataset.token2id(dataset.iid_field, session_aids.astype(str).values)),
            (0, max(pad_length - len(session_aids), 0)),
            'constant',
            0,
        ).reshape(1, -1)
        interaction = Interaction({
            'aid_list': padded_item_sequence,
            'item_length': torch.tensor([len(session_aids)])
        })

    with torch.no_grad():
        # Pass interaction from model without computing gradients and move it CPU
        scores = model.full_sort_predict(interaction.to(model.device)).detach().cpu().view(-1, dataset.item_num)

    # Set score of PAD token to -inf and retrieve top-k scores and aids
    scores[:, 0] = -np.inf
    topk_scores, topk_iids = torch.topk(scores, topk)
    prediction_scores = topk_scores.numpy()
    prediction_aids = dataset.id2token(dataset.iid_field, topk_iids).astype(int)

    return prediction_aids, prediction_scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    # Load recbole config, model and dataset
    recbole_config = yaml.load(open(settings.MODELS / args.config_path, 'r'), Loader=yaml.FullLoader)
    recbole_model_directory = (settings.MODELS / args.config_path).parents[0]
    recbole_model_path = glob.glob(f'{recbole_model_directory}/*.pth')[0]
    _, recbole_model, recbole_dataset, _, _, _ = load_data_and_model(recbole_model_path)
    logging.info(f'{recbole_model_path} is loaded')

    # Load fasttext model
    fasttext_config = yaml.load(open(settings.MODELS / 'fasttext' / 'config.yaml', 'r'), Loader=yaml.FullLoader)
    fasttext_model = fasttext.load_model(str(settings.MODELS / fasttext_config['persistence']['model_directory'] / 'fasttext.bin'))
    fasttext_embedding_dimensions = fasttext_config['model']['model_args']['dim']
    logging.info(f'{fasttext_config["persistence"]["model_directory"]}/fasttext.bin is loaded')

    if fasttext_config['nns']['nns_name'] == 'annoy':

        annoy_index = AnnoyIndex(fasttext_embedding_dimensions, metric=fasttext_config['nns']['metric'])
        # Build Annoy index from loaded model embeddings
        aid_idx = {}
        idx_aid = {}
        for idx, aid in tqdm(enumerate(fasttext_model.words), total=len(fasttext_model.words)):
            if aid == '</s>':
                continue
            else:
                annoy_index.add_item(idx, fasttext_model.get_word_vector(aid))
                aid_idx[int(aid)] = idx
                idx_aid[idx] = int(aid)

        annoy_index.build(n_trees=fasttext_config['nns']['n_trees'], n_jobs=fasttext_config['nns']['n_jobs'])
        logging.info('Finished building Annoy index')

    else:
        raise ValueError('Invalid nns_name')

    if args.mode == 'validation':

        logging.info(f'Running {recbole_config["model"]["model_name"]} model in validation mode')
        df_train_labels = pd.read_pickle(settings.DATA / 'train_labels.pkl')
        logging.info(f'Train Labels Shape: {df_train_labels.shape} - Memory Usage: {df_train_labels.memory_usage().sum() / 1024 ** 2:.2f} MB')

        # Cut session aids and event types from their cutoff index
        df_train_labels['aid'] = df_train_labels[['aid', 'session_cutoff_idx']].apply(lambda x: x['aid'][:x['session_cutoff_idx'] + 1], axis=1)
        df_train_labels['type'] = df_train_labels[['type', 'session_cutoff_idx']].apply(lambda x: x['type'][:x['session_cutoff_idx'] + 1], axis=1)

        # Specify prediction types for different models
        df_train_labels['session_unique_aid_count'] = df_train_labels['aid'].apply(lambda session_aids: len(set(session_aids)))
        df_train_labels['all_aids_in_embeddings'] = df_train_labels['aid'].apply(lambda session_aids: all([str(aid) in recbole_dataset.field2token_id['aid'] for aid in session_aids]))
        df_train_labels['session_in_embeddings'] = df_train_labels['session'].apply(lambda session: str(session) in recbole_dataset.field2token_id['session'])
        recency_weight_predictions_idx = df_train_labels['session_unique_aid_count'] >= 20
        recbole_predictions_idx = (df_train_labels['session_in_embeddings'] == True) & (df_train_labels['all_aids_in_embeddings'] == True)
        df_train_labels.loc[recency_weight_predictions_idx, 'prediction_type'] = 'recency_weight'
        df_train_labels.loc[recbole_predictions_idx, 'prediction_type'] = 'recbole'
        df_train_labels.loc[~(recency_weight_predictions_idx | recbole_predictions_idx), 'prediction_type'] = 'fasttext'
        df_train_labels.drop(columns=['all_aids_in_embeddings', 'session_in_embeddings'], inplace=True)
        del recency_weight_predictions_idx, recbole_predictions_idx
        logging.info(f'Prediction type distribution: {json.dumps(df_train_labels["prediction_type"].value_counts().to_dict(), indent=2)}')

        df_train_labels['click_predictions'] = np.nan
        df_train_labels['click_predictions'] = df_train_labels['click_predictions'].astype(object)
        df_train_labels['cart_predictions'] = np.nan
        df_train_labels['cart_predictions'] = df_train_labels['cart_predictions'].astype(object)
        df_train_labels['order_predictions'] = np.nan
        df_train_labels['order_predictions'] = df_train_labels['order_predictions'].astype(object)

        event_type_coefficient = {0: 1, 1: 6, 2: 3}
        recency_weight_predictions_idx = df_train_labels['prediction_type'] == 'recency_weight'
        for idx, row in tqdm(df_train_labels.loc[recency_weight_predictions_idx].iterrows(), total=df_train_labels.loc[recency_weight_predictions_idx].shape[0]):

            session_aids = row['aid']
            session_event_types = row['type']
            session_unique_aids = list(np.unique(session_aids))

            # Calculate click, cart and order weights based on recency
            click_weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
            cart_and_order_weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
            session_aid_click_weights = defaultdict(lambda: 0)
            session_aid_cart_and_order_weights = defaultdict(lambda: 0)

            for aid, event_type, click_weight, cart_and_order_weight in zip(session_aids, session_event_types, click_weights, cart_and_order_weights):
                session_aid_click_weights[aid] += (click_weight * event_type_coefficient[event_type])
                session_aid_cart_and_order_weights[aid] += (cart_and_order_weight * event_type_coefficient[event_type])

            # Sort click aids by their weights in descending order and take top 20 aids
            click_predictions = [aid for aid, weight in sorted(session_aid_click_weights.items(), key=lambda item: -item[1])][:20]
            # Sort cart and order aids by their weights in descending order and take top 20 aids
            cart_and_order_predictions = [aid for aid, weight in sorted(session_aid_cart_and_order_weights.items(), key=lambda item: -item[1])][:20]

            df_train_labels.at[idx, 'click_predictions'] = click_predictions
            df_train_labels.at[idx, 'cart_predictions'] = cart_and_order_predictions
            df_train_labels.at[idx, 'order_predictions'] = cart_and_order_predictions

        logging.info(f'{recency_weight_predictions_idx.sum()} sessions are predicted with recency weight')

        fasttext_predictions_idx = df_train_labels['prediction_type'] == 'fasttext'
        for idx, row in tqdm(df_train_labels.loc[fasttext_predictions_idx].iterrows(), total=df_train_labels.loc[fasttext_predictions_idx].shape[0]):

            session_aids = row['aid']
            session_unique_aids = list(np.unique(session_aids))

            # Select 20 nearest neighbors of the last session aid using FastText embeddings
            nearest_neighbors = [
                idx_aid[idx] for idx in annoy_index.get_nns_by_item(
                    i=aid_idx[session_aids[-1]],
                    n=21,
                    search_k=-1
                )[1:]
            ]

            predictions = (session_unique_aids + nearest_neighbors)[:20]
            df_train_labels.at[idx, 'click_predictions'] = predictions
            df_train_labels.at[idx, 'cart_predictions'] = predictions
            df_train_labels.at[idx, 'order_predictions'] = predictions

        logging.info(f'{fasttext_predictions_idx.sum()} sessions are predicted with fasttext')

        recbole_predictions_idx = np.where(df_train_labels['prediction_type'] == 'recbole')[0]
        recbole_predictions_batch_size = 2048
        for batch_idx in tqdm(range(len(recbole_predictions_idx) // recbole_predictions_batch_size + 1)):

            prediction_idx = recbole_predictions_idx[(batch_idx * recbole_predictions_batch_size):((batch_idx + 1) * recbole_predictions_batch_size)]
            sessions = df_train_labels.loc[prediction_idx, 'session']
            session_aids = df_train_labels.loc[prediction_idx, 'aid']
            session_unique_aids = session_aids.apply(lambda x: np.unique(x))

            prediction_aids, prediction_scores = predict(
                sessions=sessions,
                session_aids=session_aids,
                topk=20,
                pad_length=20,
                dataset=recbole_dataset,
                model=recbole_model
            )

            # Concatenate session unique aids and predictions in parallel
            session_unique_aids = pd.concat((session_unique_aids, pd.Series(np.arange(len(prediction_idx)), index=session_unique_aids.index, name='i')), axis=1)
            df_train_labels.at[
                prediction_idx, ['click_predictions', 'cart_predictions', 'order_predictions']
            ] = session_unique_aids.apply(lambda x: (x['aid'].tolist() + prediction_aids[x['i']].tolist())[:20], axis=1)

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
            {recbole_config["model"]["model_name"]} model validation scores
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

        logging.info(f'Running {recbole_config["model"]["model_name"]} model in submission mode')
        df_test = pd.read_pickle(settings.DATA / 'test.pkl')
        df_test = df_test.groupby('session')[['aid', 'type']].agg(list).reset_index()
        logging.info(f'Test Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

        # Specify prediction types for different models
        df_test['session_unique_aid_count'] = df_test['aid'].apply(lambda session_aids: len(set(session_aids)))
        df_test['all_aids_in_embeddings'] = df_test['aid'].apply(lambda session_aids: all([str(aid) in recbole_dataset.field2token_id['aid'] for aid in session_aids]))
        df_test['session_in_embeddings'] = df_test['session'].apply(lambda session: str(session) in recbole_dataset.field2token_id['session'])
        recency_weight_predictions_idx = df_test['session_unique_aid_count'] >= 20
        recbole_predictions_idx = (df_test['session_in_embeddings'] == True) & (df_test['all_aids_in_embeddings'] == True)
        df_test.loc[recbole_predictions_idx, 'prediction_type'] = 'recbole'
        df_test.loc[~(recency_weight_predictions_idx | recbole_predictions_idx), 'prediction_type'] = 'fasttext'
        df_test.loc[recency_weight_predictions_idx, 'prediction_type'] = 'recency_weight'
        df_test.drop(columns=['all_aids_in_embeddings', 'session_in_embeddings'], inplace=True)
        del recency_weight_predictions_idx, recbole_predictions_idx
        logging.info(f'Prediction type distribution: {json.dumps(df_test["prediction_type"].value_counts().to_dict(), indent=2)}')

        event_type_coefficient = {0: 1, 1: 6, 2: 3}
        recency_weight_predictions_idx = df_test['prediction_type'] == 'recency_weight'
        for idx, row in tqdm(df_test.loc[recency_weight_predictions_idx].iterrows(), total=df_test.loc[recency_weight_predictions_idx].shape[0]):

            session_aids = row['aid']
            session_event_types = row['type']
            session_unique_aids = list(np.unique(session_aids))

            # Calculate click, cart and order weights based on recency
            click_weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
            cart_and_order_weights = np.logspace(0.1, 1, len(session_aids), base=2, endpoint=True) - 1
            session_aid_click_weights = defaultdict(lambda: 0)
            session_aid_cart_and_order_weights = defaultdict(lambda: 0)

            for aid, event_type, click_weight, cart_and_order_weight in zip(session_aids, session_event_types, click_weights, cart_and_order_weights):
                session_aid_click_weights[aid] += (click_weight * event_type_coefficient[event_type])
                session_aid_cart_and_order_weights[aid] += (cart_and_order_weight * event_type_coefficient[event_type])

            # Sort click aids by their weights in descending order and take top 20 aids
            click_predictions = [aid for aid, weight in sorted(session_aid_click_weights.items(), key=lambda item: -item[1])][:20]
            # Sort cart and order aids by their weights in descending order and take top 20 aids
            cart_and_order_predictions = [aid for aid, weight in sorted(session_aid_cart_and_order_weights.items(), key=lambda item: -item[1])][:20]

            for event_type, predictions in zip(['click', 'cart', 'order'], [click_predictions, cart_and_order_predictions, cart_and_order_predictions]):
                test_predictions.append({
                    'session_type': f'{row["session"]}_{event_type}s',
                    'labels': ' '.join([str(aid) for aid in predictions])
                })

        logging.info(f'{recency_weight_predictions_idx.sum()} sessions are predicted with recency weight')

        fasttext_predictions_idx = df_test['prediction_type'] == 'fasttext'
        for idx, row in tqdm(df_test.loc[fasttext_predictions_idx].iterrows(), total=df_test.loc[fasttext_predictions_idx].shape[0]):

            session_aids = row['aid']
            session_unique_aids = list(np.unique(session_aids))

            # Select 20 nearest neighbors of the last session aid using FastText embeddings
            nearest_neighbors = [
                idx_aid[idx] for idx in annoy_index.get_nns_by_item(
                    i=aid_idx[session_aids[-1]],
                    n=21,
                    search_k=-1
                )[1:]
            ]

            predictions = (session_unique_aids + nearest_neighbors)[:20]
            for event_type, predictions in zip(['click', 'cart', 'order'], [predictions, predictions, predictions]):
                test_predictions.append({
                    'session_type': f'{row["session"]}_{event_type}s',
                    'labels': ' '.join([str(aid) for aid in predictions])
                })

        logging.info(f'{fasttext_predictions_idx.sum()} sessions are predicted with fasttext')

        recbole_predictions_idx = np.where(df_test['prediction_type'] == 'recbole')[0]
        recbole_predictions_batch_size = 4096
        for batch_idx in tqdm(range(len(recbole_predictions_idx) // recbole_predictions_batch_size + 1)):

            prediction_idx = recbole_predictions_idx[(batch_idx * recbole_predictions_batch_size):((batch_idx + 1) * recbole_predictions_batch_size)]
            sessions = df_test.loc[prediction_idx, 'session']
            session_aids = df_test.loc[prediction_idx, 'aid']
            session_unique_aids = session_aids.apply(lambda x: np.unique(x))

            prediction_aids, prediction_scores = predict(
                sessions=sessions,
                session_aids=session_aids,
                topk=20,
                pad_length=20,
                dataset=recbole_dataset,
                model=recbole_model
            )

            # Concatenate session unique aids and predictions in parallel
            session_unique_aids = pd.concat((session_unique_aids, pd.Series(np.arange(len(prediction_idx)), index=session_unique_aids.index, name='i')), axis=1)
            predictions = session_unique_aids.apply(lambda x: (x['aid'].tolist() + prediction_aids[x['i']].tolist())[:20], axis=1)

            for session, predictions_ in zip(sessions, predictions):
                for event_type, predictions__ in zip(['click', 'cart', 'order'], [predictions_, predictions_, predictions_]):
                    test_predictions.append({
                        'session_type': f'{session}_{event_type}s',
                        'labels': ' '.join([str(aid) for aid in predictions__])
                    })

        df_test_predictions = pd.DataFrame(test_predictions)
        df_test_predictions.to_csv(submissions_directory / f'{recbole_config["model"]["model_name"].lower()}_submission.csv.gz', index=False, compression='gzip')
        logging.info(f'Saved {recbole_config["model"]["model_name"].lower()}_submission.csv.gz to {submissions_directory}')

    else:
        raise ValueError('Invalid mode')
