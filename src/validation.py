import logging
from tqdm import tqdm
import numpy as np
import pandas as pd

import settings


def get_labels(aids, event_types):

    """
    Create ground-truth labels from given session aids and event types

    Parameters
    ----------
    aids: array-like of shape (n_events)
        Session aids

    event_types: array-like of shape (n_events)
        Session event types

    Returns
    -------
    labels: list of shape (n_events)
        Ground-truth labels
    """

    previous_click = None
    previous_carts = set()
    previous_orders = set()
    labels = []

    for aid, event_type in zip(reversed(aids), reversed(event_types)):

        label = {}

        if event_type == 0:
            previous_click = aid
        elif event_type == 1:
            previous_carts.add(aid)
        elif event_type == 2:
            previous_orders.add(aid)

        label[0] = previous_click
        label[1] = list(previous_carts.copy()) if len(previous_carts) > 0 else np.nan
        label[2] = list(previous_orders.copy()) if len(previous_orders) > 0 else np.nan
        labels.append(label)

    labels = labels[:-1][::-1]
    labels.append({0: [], 1: [], 2: []})

    return labels


if __name__ == '__main__':

    df_train = pd.read_pickle(settings.DATA / 'train.pkl')
    logging.info(f'Dataset Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Select sessions from last week and aggregate them as lists
    df_train_sessions = df_train.loc[df_train['session'] >= 11098528]
    df_train_sessions = df_train_sessions.groupby('session')[['aid', 'type']].agg(list).reset_index()

    df_train_sessions['click_labels'] = np.nan
    df_train_sessions['click_labels'] = df_train_sessions['click_labels'].astype(object)
    df_train_sessions['cart_labels'] = np.nan
    df_train_sessions['cart_labels'] = df_train_sessions['cart_labels'].astype(object)
    df_train_sessions['order_labels'] = np.nan
    df_train_sessions['order_labels'] = df_train_sessions['order_labels'].astype(object)

    for idx, row in tqdm(df_train_sessions.iterrows(), total=df_train_sessions.shape[0]):

        if len(row['aid']) == 2:
            # Split session from middle if there are two events
            session_cutoff_idx = 0
        else:
            # Split session from a random event but always keep at least one click at the end
            # Ground-truth labels can always have a click this way
            session_last_click_idx = np.where(np.array(row['type']) == 0)[0][-1]
            if session_last_click_idx == 0:
                session_cutoff_idx = 0
            else:
                session_cutoff_idx = np.random.randint(0, session_last_click_idx)

        df_train_sessions.at[idx, 'session_cutoff_idx'] = session_cutoff_idx

        session_labels = get_labels(aids=row['aid'], event_types=row['type'])[session_cutoff_idx]
        df_train_sessions.at[idx, 'click_labels'] = session_labels[0]
        df_train_sessions.at[idx, 'cart_labels'] = session_labels[1]
        df_train_sessions.at[idx, 'order_labels'] = session_labels[2]

    df_train_sessions = df_train_sessions.fillna(df_train_sessions.notna().applymap(lambda x: x or []))
    df_train_sessions['session_cutoff_idx'] = df_train_sessions['session_cutoff_idx'].astype(int)
    df_train_sessions.to_pickle(settings.DATA / 'train_labels.pkl')
    logging.info(f'Saved train_labels.pkl to {settings.DATA}')
