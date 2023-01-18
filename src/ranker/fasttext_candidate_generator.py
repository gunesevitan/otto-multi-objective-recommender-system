import sys
import logging
import argparse
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import polars as pl
import fasttext
from annoy import AnnoyIndex

sys.path.append('..')
import settings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    candidate_directory = pathlib.Path(settings.DATA / 'candidate')
    candidate_directory.mkdir(parents=True, exist_ok=True)

    if args.mode == 'validation':
        model_path = str(settings.MODELS / 'fasttext' / 'train_fasttext.bin')
    elif args.mode == 'submission':
        model_path = str(settings.MODELS / 'fasttext' / 'train_and_test_fasttext.bin')
    else:
        raise ValueError('Invalid mode')

    model = fasttext.load_model(model_path)
    embedding_dimensions = 32
    logging.info(f'fasttext.bin is loaded')

    annoy_index = AnnoyIndex(embedding_dimensions, metric='euclidean')
    aid_idx = {}
    idx_aid = {}
    for idx, aid in tqdm(enumerate(model.words), total=len(model.words)):
        if aid == '</s>':
            continue
        else:
            annoy_index.add_item(idx, model.get_word_vector(aid))
            aid_idx[int(aid)] = idx
            idx_aid[idx] = int(aid)

    annoy_index.build(n_trees=100, n_jobs=-1)
    logging.info('Finished building Annoy index')

    if args.mode == 'validation':

        logging.info('Running fasttext candidate generation on training set')
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

        df_val['candidates'] = np.nan
        df_val['candidates'] = df_val['candidates'].astype(object)
        df_val['candidate_scores'] = np.nan
        df_val['candidate_scores'] = df_val['candidate_scores'].astype(object)
        df_val['click_candidate_labels'] = np.nan
        df_val['click_candidate_labels'] = df_val['click_candidate_labels'].astype(object)
        df_val['cart_candidate_labels'] = np.nan
        df_val['cart_candidate_labels'] = df_val['cart_candidate_labels'].astype(object)
        df_val['order_candidate_labels'] = np.nan
        df_val['order_candidate_labels'] = df_val['order_candidate_labels'].astype(object)

        for idx, row in tqdm(df_val.iterrows(), total=df_val.shape[0]):

            session_aids = row['aid']

            # Select 100 nearest neighbors of the last session aid
            sorted_aids, sorted_aid_distances = annoy_index.get_nns_by_item(
                i=aid_idx[session_aids[-1]],
                n=21,
                search_k=-1,
                include_distances=True
            )
            sorted_aids = [idx_aid[idx] for idx in sorted_aids[1:]]
            sorted_aid_distances = sorted_aid_distances[1:]

            df_val.at[idx, 'candidates'] = sorted_aids
            df_val.at[idx, 'candidate_scores'] = sorted_aid_distances

            click_candidate_labels = [int(aid == row['click_labels']) for aid in sorted_aids]
            cart_candidate_labels = [int(aid in row['cart_labels']) for aid in sorted_aids]
            order_candidate_labels = [int(aid in row['order_labels']) for aid in sorted_aids]

            df_val.at[idx, 'click_candidate_labels'] = click_candidate_labels
            df_val.at[idx, 'cart_candidate_labels'] = cart_candidate_labels
            df_val.at[idx, 'order_candidate_labels'] = order_candidate_labels

        df_val['click_hits'] = pl.DataFrame(df_val[['candidates', 'click_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        click_recall = df_val['click_hits'].sum() / df_val['click_labels'].apply(len).clip(0, 20).sum()
        df_val['cart_hits'] = pl.DataFrame(df_val[['candidates', 'cart_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        cart_recall = df_val['cart_hits'].sum() / df_val['cart_labels'].apply(len).clip(0, 20).sum()
        df_val['order_hits'] = pl.DataFrame(df_val[['candidates', 'order_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
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

            candidate_columns = ['candidates', 'candidate_scores', f'{event_type}_candidate_labels']
            df_val_event = df_val.explode(candidate_columns)[['session'] + candidate_columns].rename(columns={
                f'{event_type}_candidate_labels': 'candidate_labels'
            })
            df_val_event = df_val_event.loc[df_val_event['candidates'].notna()]
            logging.info(
                f'''
                {event_type} fasttext candidate generation
                {df_val_event.shape[0]} candidates are generated for {df_val_event["session"].nunique()} sessions
                Candidate labels - Positives: {df_val_event["candidate_labels"].sum()} Negatives: {(df_val_event["candidate_labels"] == 0).sum()}
                Candidate scores - Positives {df_val_event.loc[df_val_event["candidate_labels"] == 1, "candidate_scores"].mean():.4f} Negatives {df_val_event.loc[df_val_event["candidate_labels"] == 0, "candidate_scores"].mean():.4f} All {df_val_event["candidate_scores"].mean():.4f}
                '''
            )
            df_val_event['candidates'] = df_val_event['candidates'].astype(np.uint64)
            df_val_event['candidate_scores'] = df_val_event['candidate_scores'].astype(np.float32)
            df_val_event['candidate_labels'] = df_val_event['candidate_labels'].astype(np.uint8)
            df_val_event.to_pickle(candidate_directory / f'{event_type}_fasttext_validation.pkl')

    elif args.mode == 'submission':

        logging.info('Running fasttext candidate generation on test set')
        df = pd.read_pickle(settings.DATA / 'test.pkl')
        df = df.groupby('session')[['aid', 'type']].agg(list).reset_index()
        logging.info(f'Test Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

        df['candidates'] = np.nan
        df['candidates'] = df['candidates'].astype(object)
        df['candidate_distances'] = np.nan
        df['candidate_distances'] = df['candidate_distances'].astype(object)

        df['click_candidate_labels'] = np.nan
        df['click_candidate_labels'] = df['click_candidate_labels'].astype(object)
        df['cart_candidate_labels'] = np.nan
        df['cart_candidate_labels'] = df['cart_candidate_labels'].astype(object)
        df['order_candidate_labels'] = np.nan
        df['order_candidate_labels'] = df['order_candidate_labels'].astype(object)

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

            session_aids = row['aid']

            # Select 100 nearest neighbors of the last session aid
            sorted_aids, sorted_aid_distances = annoy_index.get_nns_by_item(
                i=aid_idx[session_aids[-1]],
                n=101,
                search_k=-1,
                include_distances=True
            )
            sorted_aids = [idx_aid[idx] for idx in sorted_aids[1:]]
            sorted_aid_distances = sorted_aid_distances[1:]

            df.at[idx, 'candidates'] = sorted_aids
            df.at[idx, 'candidate_distances'] = sorted_aid_distances

            click_candidate_labels = [int(aid == row['click_labels']) for aid in sorted_aids]
            cart_candidate_labels = [int(aid in row['cart_labels']) for aid in sorted_aids]
            order_candidate_labels = [int(aid in row['order_labels']) for aid in sorted_aids]

            df.at[idx, 'click_candidate_labels'] = click_candidate_labels
            df.at[idx, 'cart_candidate_labels'] = cart_candidate_labels
            df.at[idx, 'order_candidate_labels'] = order_candidate_labels

        candidate_columns = [
            'candidates', 'candidate_distances',
            'click_candidate_labels', 'cart_candidate_labels', 'order_candidate_labels'
        ]

        session_chunk_size = df.shape[0] // 10
        chunks = range((df.shape[0] // session_chunk_size) + 1)

        for chunk_idx in chunks:

            session_chunk_start = chunk_idx * session_chunk_size
            session_chunk_end = (chunk_idx + 1) * session_chunk_size

            df_chunk = df.loc[session_chunk_start:session_chunk_end]

            df_chunk = df_chunk.explode(candidate_columns)[['session'] + candidate_columns]
            df_chunk['candidates'] = df_chunk['candidates'].astype(np.uint64)
            df_chunk['candidate_distances'] = df_chunk['candidate_distances'].astype(np.float32)
            df_chunk['click_candidate_labels'] = df_chunk['click_candidate_labels'].astype(np.uint8)
            df_chunk['cart_candidate_labels'] = df_chunk['cart_candidate_labels'].astype(np.uint8)
            df_chunk['order_candidate_labels'] = df_chunk['order_candidate_labels'].astype(np.uint8)

            df_chunk.to_pickle(candidate_directory / f'fasttext_validation{chunk_idx}.pkl')
            logging.info(f'fasttext_test{chunk_idx}.pkl is saved to {candidate_directory}')

        del df

        df = []
        for chunk in chunks:
            df.append(pd.read_pickle(f'fasttext_test{chunk}.pkl'))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        df.to_pickle(candidate_directory / 'fasttext_test.pkl')
        logging.info(f'fasttext_test.pkl is saved to {candidate_directory}')

    else:
        raise ValueError('Invalid mode')
