from contextlib import contextmanager
from multiprocessing.dummy import Pool
from typing import Optional

from datasets import Dataset, load_dataset

from .defs import DATASET_NAME, DEFAULT_SPLIT


def get_penguin_dataset(limit: Optional[int] = None) -> Dataset:
    """Load and return the penguin dataset.

    Args:
        limit (Optional[int]): If provided, limits the dataset to the first N entries.
            If None, returns the complete dataset.

    Returns:
        Dataset: The loaded dataset, optionally limited to specified size.
    """
    dataset = load_dataset(DATASET_NAME)
    x = dataset[DEFAULT_SPLIT]
    if limit is not None:
        x = x.select(range(limit))
    return x


@contextmanager
def get_mapper(concurrency: int):
    """Get a mapping function that optionally enables parallel processing.

    Args:
        concurrency (int): Number of concurrent processes to use.
            If 1, returns the built-in map function.
            If > 1, returns a parallel mapping function using multiprocessing.Pool.

    Yields:
        Callable: A mapping function (either built-in map or pool.imap)

    Raises:
        AssertionError: If concurrency is less than 1
    """
    assert concurrency >= 1
    if concurrency == 1:
        yield map
    else:
        with Pool(concurrency) as pool:
            yield pool.imap
