from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def apply_df(input_args):
    df, func, kwargs = input_args
    if "progress_bar" in kwargs:
        progress_bar = kwargs.pop('progress_bar')
    else:
        progress_bar = False
    if "args" in kwargs:
        args_ = kwargs.pop('args')
    else:
        args_ = None
    if progress_bar:
        tqdm.pandas(leave=False, desc=func.__name__, unit='emails',
                    dynamic_ncols=True, mininterval=2.0)
        df = df.progress_apply(func, axis=1, args=args_)
    else:
        df = df.apply(func, axis=1, args=args_)
    return df


def apply_parallel(df, func, **kwargs):
    """Apply a function along an axis of the DataFrame using multiprocessing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where the function is applied

    func : function to apply

    Returns
    -------
    pd.DataFrame
        Returns the DataFrame with the function applied.
    """
    num_workers = cpu_count()  # - 1

    if (df.shape[0] == 1) or (num_workers == 1):
        return apply_df((df, func, kwargs))

    retLst = Parallel(n_jobs=num_workers)(delayed(apply_df)(
        input_args=(d, func, kwargs)) for d in np.array_split(df, num_workers))
    return pd.concat(retLst)
