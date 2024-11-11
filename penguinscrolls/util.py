from contextlib import contextmanager
from multiprocessing.dummy import Pool
from typing import Optional

from datasets import Dataset, load_dataset

from .defs import DATASET_NAME, DEFAULT_SPLIT


def get_penguin_dataset(limit: Optional[int] = None) -> Dataset:
    dataset = load_dataset(DATASET_NAME)
    x = dataset[DEFAULT_SPLIT]
    if limit is not None:
        x = x.select(range(limit))
    return x

@contextmanager
def get_mapper(concurrency: int):
    assert concurrency >= 1
    if concurrency == 1:
        yield map
    else:
        with Pool(concurrency) as pool:
            yield pool.imap