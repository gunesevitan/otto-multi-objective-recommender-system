import sys
import logging
from tqdm import tqdm
import heapq
import numba
import gc
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


@numba.jit(nopython=True, cache=True)
def get_single_pairs(pairs, aids, timestamps, event_types, idx, length, start_time, event_weights, mode):

    """
    Get pairs from a session and update given pairs

    Parameters
    ----------
    pairs: dict
        Dictionary of pairs

    aids: numpy.ndarray of shape (223644219)
        Single array of aids

    timestamps: numpy.ndarray of shape (223644219)
        Single array of timestamps

    event_types: numpy.ndarray of shape (223644219)
        Single array of event_types

    idx: int
        Index of the event

    length: int
        Length of the event

    event_weights: numpy.ndarray of shape (3)
        Array of weights

    mode: int
        Weights mode
    """

    max_idx = idx + length
    min_idx = max(max_idx - tail, idx)

    # Iterate over the session
    for i in range(min_idx, max_idx):
        for j in range(i + 1, max_idx):

            # Take pairs which are close to each other for less than 1 day
            if timestamps[j] - timestamps[i] >= 24 * 60 * 60:
                break

            # Skip duplicate aids
            if aids[i] == aids[j]:
                continue

            if mode == 0:
                w1 = event_weights[event_types[j]]
                w2 = event_weights[event_types[i]]
            elif mode == 1:
                w1 = 1 + 3 * (timestamps[i] + start_time - 1659304800) / (1662328791 - 1659304800)
                w2 = 1 + 3 * (timestamps[j] + start_time - 1659304800) / (1662328791 - 1659304800)
            else:
                w1 = None
                w2 = None

            pairs[(aids[i], aids[j])] = w1
            pairs[(aids[j], aids[i])] = w2


@numba.jit(nopython=True, parallel=True, cache=True)
def get_pairs(aids, timestamps, event_types, row, counters, event_weights, mode):

    """
    Get pairs from each session in parallel

    Parameters
    ----------
    aids: numpy.ndarray of shape (223644219)
        Single array of aids

    timestamps: numpy.ndarray of shape (223644219)
        Single array of timestamps

    event_types: numpy.ndarray of shape (223644219)
        Single array of event_types

    row: numpy.ndarray of shape (n_events, 4)
        Events array

    counters: List
        List of counters

    event_weights: numpy.ndarray of shape (3)
        Array of weights

    mode: int
        Weights mode
    """

    par_n = len(row)
    pairs = [{(0, 0): 0.0 for _ in range(0)} for _ in range(par_n)]

    for par_i in numba.prange(par_n):
        _, idx, length, start_time = row[par_i]
        get_single_pairs(
            pairs=pairs[par_i],
            aids=aids,
            timestamps=timestamps,
            event_types=event_types,
            idx=idx,
            length=length,
            start_time=start_time,
            event_weights=event_weights,
            mode=mode
        )

    for par_i in range(par_n):
        for (aid1, aid2), w in pairs[par_i].items():

            if aid1 not in counters:
                counters[aid1] = {0: 0.0 for _ in range(0)}

            counter = counters[aid1]
            if aid2 not in counter:
                counter[aid2] = 0.0
            counter[aid2] += w


@numba.jit(nopython=True, cache=True)
def heap_topk(counter, overwrite, cap):

    """
    Get most common keys from a counter

    Parameters
    ----------
    counter: dict
        Counter of aids

    overwrite: bool
        Whether to overwrite or not

    cap: int
        Heap item limit

    Returns
    -------
    List
        Counters with top-k items
    """

    q = [(0.0, 0, 0) for _ in range(0)]
    for i, (k, n) in enumerate(counter.items()):

        if overwrite == 1:
            heapq.heappush(q, (n, i, k))
        else:
            heapq.heappush(q, (n, -i, k))

        if len(q) > cap:
            heapq.heappop(q)

    return [heapq.heappop(q)[2] for _ in range(len(q))][::-1]


@numba.jit(nopython=True, cache=True)
def get_topk(counters, topk, k):

    """
    Get top-k items from counters

    Parameters
    ----------
    counters: dict
        Dictionary of counters

    topk: dict
        Dictionary of counters with top-k items

    k: int
        Counter item count
    """

    for aid1, counter in counters.items():
        topk[aid1] = np.array(heap_topk(counter, overwrite=True, cap=k))


if __name__ == '__main__':

    dataset_directory = settings.DATA / 'carnozhao_dataset'

    df = pd.read_csv(dataset_directory / 'train.csv')
    df_test = pd.read_csv(dataset_directory / 'test.csv')
    df = pd.concat([df, df_test]).reset_index(drop=True)

    npz = np.load(dataset_directory / 'train.npz')
    npz_test = np.load(dataset_directory / 'test.npz')
    aids = np.concatenate([npz['aids'], npz_test['aids']])
    ts = np.concatenate([npz['ts'], npz_test['ts']])
    ops = np.concatenate([npz['ops'], npz_test['ops']])

    df['idx'] = np.cumsum(df['length']) - df['length']
    df['end_time'] = df['start_time'] + ts[df['idx'] + df['length'] - 1]

    logging.info(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    tail = 30
    topn = 20
    ops_weights = np.array([1.0, 6.0, 3.0])
    test_ops_weights = np.array([1.0, 6.0, 3.0])
    OP_WEIGHT = 0
    TIME_WEIGHT = 1
    parallel = 1024
    topks = {}

    for mode in [OP_WEIGHT, TIME_WEIGHT]:

        counters = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.typeof(
                numba.typed.Dict.empty(
                    key_type=numba.types.int64,
                    value_type=numba.types.float64
                )
            )
        )
        max_idx = len(df)
        for idx in tqdm(range(0, max_idx, parallel)):
            row = df.iloc[idx:min(idx + parallel, max_idx)][['session', 'idx', 'length', 'start_time']].values
            get_pairs(aids, ts, ops, row, counters, ops_weights, mode)

        topk = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.int64[:])
        get_topk(counters, topk, topn)

        del counters
        gc.collect()
        topks[mode] = topk
