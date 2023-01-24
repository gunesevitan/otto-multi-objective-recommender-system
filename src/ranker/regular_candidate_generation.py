import sys
import logging
import argparse
import pathlib
from collections import Counter
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
import polars as pl
import fasttext
from annoy import AnnoyIndex

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

    candidate_directory = pathlib.Path(settings.DATA / 'candidate')
    candidate_directory.mkdir(parents=True, exist_ok=True)
    covisitation_directory = pathlib.Path(settings.DATA / 'covisitation')

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

    del model
    annoy_index.build(n_trees=100, n_jobs=-1)
    logging.info('Finished building Annoy index')

    if args.mode == 'validation':

        top_time_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_time_weighted_0.pqt')))
        for i in range(1, 6):
            top_time_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_time_weighted_{i}.pqt'))))

        top_click_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_click_weighted_0.pqt')))
        for i in range(1, 6):
            top_click_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_click_weighted_{i}.pqt'))))

        top_cart_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_cart_weighted_0.pqt')))
        for i in range(1, 6):
            top_cart_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_cart_weighted_{i}.pqt'))))

        top_order_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_order_weighted_0.pqt')))
        for i in range(1, 6):
            top_order_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_order_weighted_{i}.pqt'))))

        top_click_cart_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_click_cart_0.pqt')))
        for i in range(1, 6):
            top_click_cart_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_click_cart_{i}.pqt'))))

        top_click_order_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_click_order_0.pqt')))
        for i in range(1, 6):
            top_click_order_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_click_order_{i}.pqt'))))

        top_cart_order_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / 'top_cart_order_0.pqt')))
        for i in range(1, 2):
            top_cart_order_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'validation' / f'top_cart_order_{i}.pqt'))))

        logging.info(f'Loaded top covisitation statistics for training and validation')
        logging.info('Running candidate generation in validation mode')

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

        for t in tqdm(df_val.itertuples(), total=df_val.shape[0]):

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

            # Get most similar aids of the last aid in the session
            fasttext_nearest_neighbor_idx = annoy_index.get_nns_by_item(i=aid_idx[session_aids[-1]], n=51, search_k=-1, include_distances=False)
            fasttext_similar_aids = [idx_aid[idx] for idx in fasttext_nearest_neighbor_idx[1:]]

            # Concatenate all generated click aids and select most common ones
            covisited_click_aids = time_weighted_covisited_aids + click_weighted_covisited_aids + cart_weighted_covisited_aids + click_cart_covisited_aids + cart_order_covisited_aids + fasttext_similar_aids
            sorted_click_aids = [(aid, weight) for aid, weight in Counter(covisited_click_aids).most_common(100) if aid not in session_unique_aids]
            sorted_click_aid_weights = np.arange(1, len(session_unique_aids) + 1).tolist()[::-1] + [weight for _, weight in sorted_click_aids]
            sorted_click_aids = [aid for aid, _ in sorted_click_aids]

            # Concatenate all generated cart aids and select most common ones
            covisited_cart_aids = time_weighted_covisited_aids + cart_weighted_covisited_aids + cart_order_covisited_aids + fasttext_similar_aids
            sorted_cart_aids = [(aid, weight) for aid, weight in Counter(covisited_cart_aids).most_common(100) if aid not in session_unique_aids]
            sorted_cart_aid_weights = np.arange(1, len(session_unique_aids) + 1).tolist()[::-1] + [weight for _, weight in sorted_cart_aids]
            sorted_cart_aids = [aid for aid, _ in sorted_cart_aids]

            # Concatenate all generated order aids and select most common ones
            covisited_order_aids = time_weighted_covisited_aids + cart_weighted_covisited_aids + cart_order_covisited_aids + fasttext_similar_aids
            sorted_order_aids = [(aid, weight) for aid, weight in Counter(covisited_order_aids).most_common(100) if aid not in session_unique_aids]
            sorted_order_aid_weights = np.arange(1, len(session_unique_aids) + 1).tolist()[::-1] + [weight for _, weight in sorted_order_aids]
            sorted_order_aids = [aid for aid, _ in sorted_order_aids]

            click_predictions = session_unique_aids + sorted_click_aids
            cart_predictions = session_unique_aids + sorted_cart_aids
            order_predictions = session_unique_aids + sorted_order_aids

            df_val.at[t.Index, 'click_candidates'] = click_predictions
            df_val.at[t.Index, 'cart_candidates'] = cart_predictions
            df_val.at[t.Index, 'order_candidates'] = order_predictions

            df_val.at[t.Index, 'click_candidate_scores'] = sorted_click_aid_weights
            df_val.at[t.Index, 'cart_candidate_scores'] = sorted_cart_aid_weights
            df_val.at[t.Index, 'order_candidate_scores'] = sorted_order_aid_weights

            # Create candidate labels for clicks, carts and orders
            sorted_click_aid_labels = [int(aid == t.click_labels) for aid in click_predictions]
            sorted_cart_aid_labels = [int(aid in t.cart_labels) for aid in cart_predictions]
            sorted_order_aid_labels = [int(aid in t.order_labels) for aid in order_predictions]

            df_val.at[t.Index, 'click_candidate_labels'] = sorted_click_aid_labels
            df_val.at[t.Index, 'cart_candidate_labels'] = sorted_cart_aid_labels
            df_val.at[t.Index, 'order_candidate_labels'] = sorted_order_aid_labels

        del annoy_index
        del top_time_weighted_covisitation, top_click_weighted_covisitation, top_cart_weighted_covisitation
        del top_order_weighted_covisitation, top_click_cart_covisitation, top_click_order_covisitation, top_cart_order_covisitation

        df_val['click_hits'] = pl.DataFrame(df_val[['click_candidates', 'click_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        df_val['click_hits'] = df_val['click_hits'].astype(np.uint8)
        click_recall = df_val['click_hits'].sum() / df_val['click_labels'].apply(len).clip(0, 20).sum()
        df_val['cart_hits'] = pl.DataFrame(df_val[['cart_candidates', 'cart_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        df_val['cart_hits'] = df_val['cart_hits'].astype(np.uint8)
        cart_recall = df_val['cart_hits'].sum() / df_val['cart_labels'].apply(len).clip(0, 20).sum()
        df_val['order_hits'] = pl.DataFrame(df_val[['order_candidates', 'order_labels']]).apply(lambda x: len(set(x[0]).intersection(set(x[1])))).to_pandas().values.reshape(-1)
        df_val['order_hits'] = df_val['order_hits'].astype(np.uint8)
        order_recall = df_val['order_hits'].sum() / df_val['order_labels'].apply(len).clip(0, 20).sum()
        weighted_recall = (click_recall * 0.1) + (cart_recall * 0.3) + (order_recall * 0.6)
        df_val.drop(columns=['click_hits', 'cart_hits', 'order_hits'], inplace=True)

        logging.info(
            f'''
            Candidate max recalls
            Click max recall: {click_recall:.6f}
            Cart max recall: {cart_recall:.6f}
            Order max recall: {order_recall:.6f}
            Weighted max recall: {weighted_recall:.6f}
            '''
        )

        # Divide validation set into equal number of chunks of sessions
        session_chunk_size = df_val.shape[0] // 15
        chunks = range((df_val.shape[0] // session_chunk_size) + 1)

        for event_type in ['click', 'cart', 'order']:

            # Select and process candidates of one event type at a time
            candidate_columns = [f'{event_type}_candidates', f'{event_type}_candidate_scores', f'{event_type}_candidate_labels']

            for chunk_idx in chunks:

                session_chunk_start = chunk_idx * session_chunk_size
                session_chunk_end = (chunk_idx + 1) * session_chunk_size

                df_chunk = df_val.loc[session_chunk_start:session_chunk_end, ['session'] + candidate_columns]
                df_chunk = df_chunk.explode(candidate_columns)[['session'] + candidate_columns].rename(columns={
                    f'{event_type}_candidates': 'candidates',
                    f'{event_type}_candidate_scores': 'candidate_scores',
                    f'{event_type}_candidate_labels': 'candidate_labels'
                })
                df_chunk = df_chunk.loc[df_chunk['candidates'].notna()]
                df_chunk['candidates'] = df_chunk['candidates'].astype(np.uint64)
                df_chunk['candidate_scores'] = df_chunk['candidate_scores'].astype(np.float32)
                df_chunk['candidate_labels'] = df_chunk['candidate_labels'].astype(np.uint8)

                df_chunk.to_pickle(candidate_directory / f'{event_type}_validation{chunk_idx}.pkl')
                logging.info(f'{event_type}_validation{chunk_idx}.pkl is saved to {candidate_directory}')

            df_candidate = []
            for chunk in chunks:
                df_candidate.append(pd.read_pickle(candidate_directory / f'{event_type}_validation{chunk}.pkl'))
            df_candidate = pd.concat(df_candidate, axis=0, ignore_index=True).reset_index(drop=True)
            df_candidate.to_pickle(candidate_directory / f'{event_type}_validation.pkl')

            logging.info(
                f'''
                {event_type} candidate generation
                {df_candidate.shape[0]} candidates are generated for {df_candidate["session"].nunique()} sessions
                Candidate labels - Positives: {df_candidate["candidate_labels"].sum()} Negatives: {(df_candidate["candidate_labels"] == 0).sum()}
                Candidate scores - Positives {df_candidate.loc[df_candidate["candidate_labels"] == 1, "candidate_scores"].mean():.4f} Negatives {df_candidate.loc[df_candidate["candidate_labels"] == 0, "candidate_scores"].mean():.4f} All {df_candidate["candidate_scores"].mean():.4f}
                '''
            )

    elif args.mode == 'submission':

        top_time_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_time_weighted_0.pqt')))
        for i in range(1, 6):
            top_time_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_time_weighted_{i}.pqt'))))

        top_click_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_click_weighted_0.pqt')))
        for i in range(1, 6):
            top_click_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_click_weighted_{i}.pqt'))))

        top_cart_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_cart_weighted_0.pqt')))
        for i in range(1, 6):
            top_cart_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_cart_weighted_{i}.pqt'))))

        top_order_weighted_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_order_weighted_0.pqt')))
        for i in range(1, 6):
            top_order_weighted_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_order_weighted_{i}.pqt'))))

        top_click_cart_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_click_cart_0.pqt')))
        for i in range(1, 6):
            top_click_cart_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_click_cart_{i}.pqt'))))

        top_click_order_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_click_order_0.pqt')))
        for i in range(1, 6):
            top_click_order_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_click_order_{i}.pqt'))))

        top_cart_order_covisitation = covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / 'top_cart_order_0.pqt')))
        for i in range(1, 2):
            top_cart_order_covisitation.update(covisitation_df_to_dict(pd.read_parquet(str(covisitation_directory / 'submission' / f'top_cart_order_{i}.pqt'))))

        logging.info(f'Loaded top covisitation statistics for entire dataset')
        logging.info('Running covisitation model in submission mode')

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

        for t in tqdm(df_test.itertuples(), total=df_test.shape[0]):

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

            # Get most similar aids of the last aid in the session
            fasttext_nearest_neighbor_idx = annoy_index.get_nns_by_item(i=aid_idx[session_aids[-1]], n=46, search_k=-1, include_distances=False)
            fasttext_similar_aids = [idx_aid[idx] for idx in fasttext_nearest_neighbor_idx[1:]]

            # Concatenate all generated click aids and select most common ones
            covisited_click_aids = time_weighted_covisited_aids + click_weighted_covisited_aids + cart_weighted_covisited_aids + click_cart_covisited_aids + cart_order_covisited_aids + fasttext_similar_aids
            sorted_click_aids = [(aid, weight) for aid, weight in Counter(covisited_click_aids).most_common(100) if aid not in session_unique_aids]
            sorted_click_aid_weights = ([0] * len(session_unique_aids)) + [weight for _, weight in sorted_click_aids]
            sorted_click_aids = [aid for aid, _ in sorted_click_aids]

            # Concatenate all generated cart aids and select most common ones
            covisited_cart_aids = time_weighted_covisited_aids + cart_weighted_covisited_aids + cart_order_covisited_aids + fasttext_similar_aids
            sorted_cart_aids = [(aid, weight) for aid, weight in Counter(covisited_cart_aids).most_common(100) if aid not in session_unique_aids]
            sorted_cart_aid_weights = ([0] * len(session_unique_aids)) + [weight for _, weight in sorted_cart_aids]
            sorted_cart_aids = [aid for aid, _ in sorted_cart_aids]

            # Concatenate all generated order aids and select most common ones
            covisited_order_aids = time_weighted_covisited_aids + cart_weighted_covisited_aids + cart_order_covisited_aids + fasttext_similar_aids
            sorted_order_aids = [(aid, weight) for aid, weight in Counter(covisited_order_aids).most_common(100) if aid not in session_unique_aids]
            sorted_order_aid_weights = ([0] * len(session_unique_aids)) + [weight for _, weight in sorted_order_aids]
            sorted_order_aids = [aid for aid, _ in sorted_order_aids]

            click_predictions = session_unique_aids + sorted_click_aids
            cart_predictions = session_unique_aids + sorted_cart_aids
            order_predictions = session_unique_aids + sorted_order_aids

            df_test.at[t.Index, 'click_candidates'] = click_predictions
            df_test.at[t.Index, 'cart_candidates'] = cart_predictions
            df_test.at[t.Index, 'order_candidates'] = order_predictions

            df_test.at[t.Index, 'click_candidate_scores'] = sorted_click_aid_weights
            df_test.at[t.Index, 'cart_candidate_scores'] = sorted_cart_aid_weights
            df_test.at[t.Index, 'order_candidate_scores'] = sorted_order_aid_weights

        del annoy_index
        del top_time_weighted_covisitation, top_click_weighted_covisitation, top_cart_weighted_covisitation
        del top_order_weighted_covisitation, top_click_cart_covisitation, top_click_order_covisitation, top_cart_order_covisitation

        # Divide test set into equal number of chunks of sessions
        session_chunk_size = df_test.shape[0] // 15
        chunks = range((df_test.shape[0] // session_chunk_size) + 1)

        for event_type in ['click', 'cart', 'order']:

            # Select and process candidates of one event type at a time
            candidate_columns = [f'{event_type}_candidates', f'{event_type}_candidate_scores']

            for chunk_idx in chunks:
                session_chunk_start = chunk_idx * session_chunk_size
                session_chunk_end = (chunk_idx + 1) * session_chunk_size

                df_chunk = df_test.loc[session_chunk_start:session_chunk_end, ['session'] + candidate_columns]
                df_chunk = df_chunk.explode(candidate_columns)[['session'] + candidate_columns].rename(columns={
                    f'{event_type}_candidates': 'candidates',
                    f'{event_type}_candidate_scores': 'candidate_scores'
                })
                df_chunk = df_chunk.loc[df_chunk['candidates'].notna()]
                df_chunk['candidates'] = df_chunk['candidates'].astype(np.uint64)
                df_chunk['candidate_scores'] = df_chunk['candidate_scores'].astype(np.float32)

                df_chunk.to_pickle(candidate_directory / f'{event_type}_test{chunk_idx}.pkl')
                logging.info(f'{event_type}_test{chunk_idx}.pkl is saved to {candidate_directory}')

            df_candidate = []
            for chunk in chunks:
                df_candidate.append(pd.read_pickle(candidate_directory / f'{event_type}_test{chunk}.pkl'))
            df_candidate = pd.concat(df_candidate, axis=0, ignore_index=True).reset_index(drop=True)
            df_candidate.to_pickle(candidate_directory / f'{event_type}_test.pkl')

            logging.info(
                f'''
                {event_type} candidate generation
                {df_candidate.shape[0]} candidates are generated for {df_candidate["session"].nunique()} sessions
                Candidate scores - {df_candidate["candidate_scores"].mean():.4f}
                '''
            )

    else:
        raise ValueError('Invalid mode')
