import sys
import logging
import argparse
import yaml
import pathlib
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import fasttext
from annoy import AnnoyIndex

sys.path.append('..')
import settings
import metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(settings.MODELS / args.config_path, 'r'), Loader=yaml.FullLoader)

    # Load Word2vec or FastText model from model directory specified in config file
    if config['model']['model_name'] == 'Word2Vec':
        model = Word2Vec.load(str(settings.MODELS / config['persistence']['model_directory'] / 'word2vec.model'))
        embedding_dimensions = config['model']['model_args']['vector_size']
        logging.info(f'{config["persistence"]["model_directory"]}/word2vec.model is loaded')
    elif config['model']['model_name'] == 'FastText':
        model = fasttext.load_model(str(settings.MODELS / config['persistence']['model_directory'] / 'fasttext.bin'))
        embedding_dimensions = config['model']['model_args']['dim']
        logging.info(f'{config["persistence"]["model_directory"]}/fasttext.bin is loaded')
    else:
        raise ValueError('Invalid model_name')

    if config['nns']['nns_name'] == 'annoy':

        annoy_index = AnnoyIndex(embedding_dimensions, metric=config['nns']['metric'])
        # Build Annoy index from loaded model embeddings
        if config['model']['model_name'] == 'Word2Vec':
            aid_idx = {aid: idx for idx, aid in enumerate(model.wv.index_to_key)}
            for aid, idx in tqdm(aid_idx.items(), total=len(aid_idx)):
                annoy_index.add_item(idx, model.wv.vectors[idx])
        elif config['model']['model_name'] == 'FastText':
            aid_idx = {}
            idx_aid = {}
            for idx, aid in tqdm(enumerate(model.words), total=len(model.words)):
                if aid == '</s>':
                    continue
                else:
                    annoy_index.add_item(idx, model.get_word_vector(aid))
                    aid_idx[int(aid)] = idx
                    idx_aid[idx] = int(aid)
        else:
            raise ValueError('Invalid model_name')

        annoy_index.build(n_trees=config['nns']['n_trees'], n_jobs=config['nns']['n_jobs'])
        logging.info('Finished building Annoy index')

    else:
        raise ValueError('Invalid nns_name')

    if args.mode == 'validation':

        logging.info(f'Running {config["model"]["model_name"]} model in validation mode')
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
                # Sort cart and order aids by their weights in descending order and take top 20 aids
                sorted_cart_and_order_aids = [aid for aid, weight in sorted(session_aid_cart_and_order_weights.items(), key=lambda item: -item[1])][:20]

                click_predictions = sorted_click_aids
                cart_predictions = sorted_cart_and_order_aids
                order_predictions = sorted_cart_and_order_aids

                df_train_labels.at[idx, 'click_predictions'] = click_predictions
                df_train_labels.at[idx, 'cart_predictions'] = cart_predictions
                df_train_labels.at[idx, 'order_predictions'] = order_predictions

            else:
                if config['model']['model_name'] == 'Word2Vec':
                    nearest_neighbors = [
                        model.wv.index_to_key[idx] for idx in annoy_index.get_nns_by_item(
                            i=aid_idx[session_aids[-1]],
                            n=21,
                            search_k=-1
                        )[1:]
                    ]
                elif config['model']['model_name'] == 'FastText':
                    if config['nns']['recursive_nns']:
                        # Select 20 nearest neighbors recursively
                        nearest_neighbors = []
                        current_aid = session_aids[-1]
                        for i in range(20 - len(session_unique_aids)):
                            next_aids = nearest_neighbors = [
                                idx_aid[idx] for idx in annoy_index.get_nns_by_item(
                                    i=aid_idx[session_aids[-1]],
                                    n=i + 2,
                                    search_k=-1
                                )[1:]
                            ]
                            for next_aid in next_aids:
                                if next_aid in (nearest_neighbors + session_unique_aids):
                                    continue
                                else:
                                    nearest_neighbors.append(next_aid)
                                    current_aid = next_aid
                    else:
                        # Select 20 nearest neighbors of the last session aid
                        nearest_neighbors = [
                            idx_aid[idx] for idx in annoy_index.get_nns_by_item(
                                i=aid_idx[session_aids[-1]],
                                n=21,
                                search_k=-1
                            )[1:]
                        ]
                else:
                    raise ValueError('Invalid model_name')

                predictions = session_unique_aids + nearest_neighbors
                df_train_labels.at[idx, 'click_predictions'] = predictions[:20]
                df_train_labels.at[idx, 'cart_predictions'] = predictions[:20]
                df_train_labels.at[idx, 'order_predictions'] = predictions[:20]

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
            {config["model"]["model_name"]} model validation scores
            clicks - n: {df_train_labels["click_recall"].notna().sum()} recall@20: {mean_click_recall:.4f}
            carts - n: {df_train_labels["cart_recall"].notna().sum()} recall@20: {mean_cart_recall:.4f}
            orders - n: {df_train_labels["order_recall"].notna().sum()} recall@20: {mean_order_recall:.4f}
            weighted recall@20: {mean_weighted_recall:.4f}
            '''
        )

    elif args.mode == 'submission':

        logging.info(f'Running {config["model"]["model_name"]} model in submission mode')
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
                    session_aid_click_weights[aid] += (click_weight * event_type_coefficient[event_type])
                    session_aid_cart_and_order_weights[aid] += (cart_and_order_weight * event_type_coefficient[event_type])

                # Sort click aids by their weights in descending order and take top 20 aids
                sorted_click_aids = [aid for aid, weight in sorted(session_aid_click_weights.items(), key=lambda item: -item[1])][:20]
                # Sort cart and order aids by their weights in descending order and take top 20 aids
                sorted_cart_and_order_aids = [aid for aid, weight in sorted(session_aid_cart_and_order_weights.items(), key=lambda item: -item[1])][:20]

                click_predictions = sorted_click_aids
                cart_predictions = sorted_cart_and_order_aids
                order_predictions = sorted_cart_and_order_aids

            else:
                if config['model']['model_name'] == 'Word2Vec':
                    nearest_neighbors = [
                        model.wv.index_to_key[idx] for idx in annoy_index.get_nns_by_item(
                            i=aid_idx[session_aids[-1]],
                            n=21,
                            search_k=-1
                        )[1:]
                    ]
                elif config['model']['model_name'] == 'FastText':
                    if config['nns']['recursive_nns']:
                        # Select 20 nearest neighbors recursively
                        nearest_neighbors = []
                        current_aid = session_aids[-1]

                        for i in range(20 - len(session_unique_aids)):
                            next_aids = nearest_neighbors = [
                                idx_aid[idx] for idx in annoy_index.get_nns_by_item(
                                    i=aid_idx[session_aids[-1]],
                                    n=i + 2,
                                    search_k=-1
                                )[1:]
                            ]
                            for next_aid in next_aids:
                                if next_aid in (nearest_neighbors + session_unique_aids):
                                    continue
                                else:
                                    nearest_neighbors.append(next_aid)
                                    current_aid = next_aid
                    else:
                        # Select 20 nearest neighbors of the last session aid
                        nearest_neighbors = [
                            idx_aid[idx] for idx in annoy_index.get_nns_by_item(
                                i=aid_idx[session_aids[-1]],
                                n=21,
                                search_k=-1
                            )[1:]
                        ]
                else:
                    raise ValueError('Invalid model_name')

                predictions = session_unique_aids + nearest_neighbors
                click_predictions = predictions
                cart_predictions = predictions
                order_predictions = predictions

            for event_type, predictions in zip(['click', 'cart', 'order'], [click_predictions, cart_predictions, order_predictions]):
                test_predictions.append({
                    'session_type': f'{row["session"]}_{event_type}s',
                    'labels': ' '.join([str(aid) for aid in predictions[:20]])
                })

        df_test_predictions = pd.DataFrame(test_predictions)
        df_test_predictions.to_csv(submissions_directory / f'{config["model"]["model_name"].lower()}_submission.csv.gz', index=False, compression='gzip')
        logging.info(f'Saved {config["model"]["model_name"].lower()}_submission.csv.gz to {submissions_directory}')

    else:
        raise ValueError('Invalid mode')
