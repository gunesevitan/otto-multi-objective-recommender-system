import sys
import logging
import argparse
from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('..')
import settings
import metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    if args.mode == 'validation':

        logging.info(f'Running TF-IDF similarity model in validation mode')
        df_train_labels = pd.read_pickle(settings.DATA / 'train_labels.pkl')
        logging.info(f'Train Labels Shape: {df_train_labels.shape} - Memory Usage: {df_train_labels.memory_usage().sum() / 1024 ** 2:.2f} MB')

        # Cut session aids and event types from their cutoff index
        df_train_labels['aid'] = df_train_labels[['aid', 'session_cutoff_idx']].apply(lambda x: x['aid'][:x['session_cutoff_idx'] + 1], axis=1)
        df_train_labels['type'] = df_train_labels[['type', 'session_cutoff_idx']].apply(lambda x: x['type'][:x['session_cutoff_idx'] + 1], axis=1)

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectors = tfidf_vectorizer.fit_transform(df_train_labels['aid'].apply(lambda x: ' '.join([str(aid) for aid in x])))
        aid2idx = {int(aid): idx for aid, idx in tfidf_vectorizer.vocabulary_.items()}
        idx2aid = {idx: int(aid) for aid, idx in tfidf_vectorizer.vocabulary_.items()}
        similarity_matrix = cosine_similarity(tfidf_vectors, dense_output=False)

        # Specify prediction types for different models
        df_train_labels['session_unique_aid_count'] = df_train_labels['aid'].apply(lambda session_aids: len(set(session_aids)))
        recency_weight_predictions_idx = df_train_labels['session_unique_aid_count'] >= 20
        df_train_labels.loc[recency_weight_predictions_idx, 'prediction_type'] = 'recency_weight'
        df_train_labels.loc[~recency_weight_predictions_idx, 'prediction_type'] = 'tfidf'
        del recency_weight_predictions_idx
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

        tfidf_predictions_idx = df_train_labels['prediction_type'] == 'tfidf'
        for idx, row in tqdm(df_train_labels.loc[tfidf_predictions_idx].iterrows(), total=df_train_labels.loc[tfidf_predictions_idx].shape[0]):

            session_aids = row['aid']
            session_unique_aids = list(np.unique(session_aids))

            similarities = similarity_matrix[aid2idx[session_aids[-1]]].toarray().reshape(-1)
            sorted_similarity_idx = np.argsort(similarities)[::-1][1:50]
            similar_aids = [idx2aid[idx_] for idx_ in sorted_similarity_idx if idx_ in idx2aid]

            predictions = (session_unique_aids + similar_aids)[:20]
            df_train_labels.at[idx, 'click_predictions'] = predictions
            df_train_labels.at[idx, 'cart_predictions'] = predictions
            df_train_labels.at[idx, 'order_predictions'] = predictions

        logging.info(f'{tfidf_predictions_idx.sum()} sessions are predicted with fasttext')

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
            TF-IDF similarity model validation scores
            clicks - n: {df_train_labels["click_recall"].notna().sum()} recall@20: {mean_click_recall:.4f}
            carts - n: {df_train_labels["cart_recall"].notna().sum()} recall@20: {mean_cart_recall:.4f}
            orders - n: {df_train_labels["order_recall"].notna().sum()} recall@20: {mean_order_recall:.4f}
            weighted recall@20: {mean_weighted_recall:.4f}
            '''
        )
