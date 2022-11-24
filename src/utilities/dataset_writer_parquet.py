import sys
import pathlib
from tqdm import tqdm
import pandas as pd

sys.path.append('..')
import settings


def ground_truth(events):

    """
    Create ground-truth from given events

    Parameters
    ----------
    events: list of shape (n_events)
        List of dictionary of events

    Returns
    -------
    events: list of shape (n_events)
        List of dictionary of events with ground-truth
    """

    prev_labels = {'clicks': None, 'carts': set(), 'orders': set()}

    for event in reversed(events):
        event['labels'] = {}

        for label in ['clicks', 'carts', 'orders']:
            if prev_labels[label]:
                if label != 'clicks':
                    event['labels'][label] = prev_labels[label].copy()
                else:
                    event['labels'][label] = prev_labels[label]

        if event['type'] == 'clicks':
            prev_labels['clicks'] = event['aid']
        if event['type'] == 'carts':
            prev_labels['carts'].add(event['aid'])
        elif event['type'] == 'orders':
            prev_labels['orders'].add(event['aid'])

    return events[:-1]


def jsonl_to_dataframe(jsonl_file_path, total=1290, test=False):

    """
    Convert jsonl format dataset into a pandas.DataFrame

    Parameters
    ----------
    jsonl_file_path: str
        Path to jsonl file

    total: int
        Number of iterations for progress bar

    test: bool
        Whether dataset is test or not

    Returns
    -------
    df: pandas.DataFrame of shape (n_events, 7)
        Dataframe with ground-truth
    """

    chunks = pd.read_json(jsonl_file_path, lines=True, chunksize=10000)
    sessions, aids, timestamps, event_types, clicks, carts, orders = [], [], [], [], [], [], []

    for chunk in tqdm(chunks, total=total):
        for row_idx, session_data in chunk.iterrows():

            aids_, timestamps_, event_types_, clicks_, carts_, orders_ = [], [], [], [], [], []

            if len(session_data['events']) > 1 and not test:
                events = ground_truth(session_data.events)
                for event in events:
                    aids_.append(event['aid'])
                    timestamps_.append(event['ts'])
                    event_types_.append(event['type'])
                    clicks_.append(event['labels'].get('clicks', None))
                    carts_.append(list(event['labels'].get('carts', [])))
                    orders_.append(list(event['labels'].get('orders', [])))
            else:
                for event in session_data.events:
                    aids_.append(event['aid'])
                    timestamps_.append(event['ts'])
                    event_types_.append(event['type'])

            sessions.append(session_data.session)
            aids.append(aids_)
            timestamps.append(timestamps_)
            event_types.append(event_types_)
            clicks.append(clicks_)
            carts.append(carts_)
            orders.append(orders_)

    df = pd.DataFrame(data={
        'session': sessions,
        'aid': aids,
        'ts': timestamps,
        'type': event_types,
        'clicks': clicks,
        'carts': carts,
        'orders': orders,
    })
    df['target'] = df['type'].apply(lambda x: [['', 'clicks', 'carts', 'orders'].index(c) for c in x])

    return df


def jsonl_to_parquet(jsonl_file_path, total=1290, test=False):

    """
    Convert jsonl format dataset into a pandas.DataFrame and save chunks as parquet files

    Parameters
    ----------
    jsonl_file_path: str
        Path to jsonl file

    total: int
        Number of iterations for progress bar

    test: bool
        Whether dataset is test or not
    """

    parquet_dataset_directory = pathlib.Path(settings.DATA / 'parquet_dataset')
    parquet_dataset_directory.mkdir(parents=True, exist_ok=True)

    chunks = pd.read_json(jsonl_file_path, lines=True, chunksize=200000)

    for i, chunk in tqdm(enumerate(chunks), total=total):

        sessions, aids, timestamps, event_types, clicks, carts, orders = [], [], [], [], [], [], []

        for row_idx, session_data in chunk.iterrows():

            aids_, timestamps_, event_types_, clicks_, carts_, orders_ = [], [], [], [], [], []

            if len(session_data['events']) > 1 and not test:
                events = ground_truth(session_data.events)
                for event in events:
                    aids_.append(event['aid'])
                    timestamps_.append(event['ts'])
                    event_types_.append(event['type'])
                    clicks_.append(event['labels'].get('clicks', None))
                    carts_.append(list(event['labels'].get('carts', [])))
                    orders_.append(list(event['labels'].get('orders', [])))
            else:
                for event in session_data.events:
                    aids_.append(event['aid'])
                    timestamps_.append(event['ts'])
                    event_types_.append(event['type'])

            sessions.append(session_data.session)
            aids.append(aids_)
            timestamps.append(timestamps_)
            event_types.append(event_types_)
            clicks.append(clicks_)
            carts.append(carts_)
            orders.append(orders_)

        df = pd.DataFrame(data={
            'session': sessions,
            'aid': aids,
            'ts': timestamps,
            'type': event_types,
            'clicks': clicks,
            'carts': carts,
            'orders': orders,
        })
        df['target'] = df['type'].apply(lambda x: [['', 'clicks', 'carts', 'orders'].index(c) for c in x])
        df.to_parquet(parquet_dataset_directory / f'train_{i}.parquet', index=False)


if __name__ == '__main__':

    jsonl_to_parquet(
        jsonl_file_path=settings.DATA / 'train.jsonl',
        total=1290,
        test=False
    )
