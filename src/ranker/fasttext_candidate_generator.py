import sys
import logging
import argparse
import pathlib
from tqdm import tqdm
import numpy as np
import pandas as pd
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

    model = fasttext.load_model(str(settings.MODELS / 'fasttext' / 'fasttext.bin'))
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
        df = pd.read_pickle(settings.DATA / 'train_labels.pkl')
        logging.info(f'Train Labels Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

        # Cut session aids and event types from their cutoff index
        df['aid'] = df[['aid', 'session_cutoff_idx']].apply(lambda x: x['aid'][:x['session_cutoff_idx'] + 1], axis=1)
        df['type'] = df[['type', 'session_cutoff_idx']].apply(lambda x: x['type'][:x['session_cutoff_idx'] + 1], axis=1)

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
            logging.info(f'fasttext_validation{chunk_idx}.pkl is saved to {candidate_directory}')

        del df

        df = []
        for chunk in chunks:
            df.append(pd.read_pickle(f'fasttext_validation{chunk}.pkl'))
        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        df.to_pickle(candidate_directory / 'fasttext_validation.pkl')
        logging.info(f'fasttext_validation.pkl is saved to {candidate_directory}')

        logging.info(
            f'''
            Fasttext candidate generation
            {df.shape[0]} candidates are generated for {df["session"].nunique()} sessions
            Candidate labels
            Click candidates - Positives: {df["click_candidate_labels"].sum()} Negatives: {(df["click_candidate_labels"] == 0).sum()}
            Cart candidates - Positives: {df["cart_candidate_labels"].sum()} Negatives: {(df["cart_candidate_labels"] == 0).sum()}
            Order candidates - Positives: {df["order_candidate_labels"].sum()} Negatives: {(df["order_candidate_labels"] == 0).sum()}
            Candidate distances
            Click candidates - Positives {df.loc[df["click_candidate_labels"] == 1, "candidate_distances"].mean():.4f} Negatives {df.loc[df["click_candidate_labels"] == 0, "candidate_distances"].mean():.4f} All {df["candidate_distances"].mean():.4f}
            Cart candidates - Positives {df.loc[df["cart_candidate_labels"] == 1, "candidate_distances"].mean():.4f} Negatives {df.loc[df["cart_candidate_labels"] == 0, "candidate_distances"].mean():.4f} All {df["candidate_distances"].mean():.4f}
            Order candidates - Positives {df.loc[df["order_candidate_labels"] == 1, "candidate_distances"].mean():.4f} Negatives {df.loc[df["order_candidate_labels"] == 0, "candidate_distances"].mean():.4f} All {df["candidate_distances"].mean():.4f}
            '''
        )

    elif args.mode == 'test':

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
