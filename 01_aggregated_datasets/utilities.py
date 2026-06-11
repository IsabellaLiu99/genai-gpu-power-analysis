import pandas as pd
import json
from typing import Union, Any, Dict
from pathlib import Path


def filter_df(df, field, value, reset_index=True, convert2float=False):
    # loop through
    for k in range(len(field)):
        # temporary values
        f = field[k]
        v = value[k]

        # check if field is within df
        e_flag = f in df.columns

        # throw error if not
        if not e_flag:
            raise ValueError("'{:s}' is not a field in the dataframe".format(str(f)))

        # convert to float (if applicable)
        if convert2float:
            v = float(v)

            try:
                # tolerance
                tol = 1e-3

                # filter
                df = df[abs(df[f] - v) < tol]

            except Exception:
                # filter
                df = df[df[f] == v]

        else:
            # filter
            if not isinstance(v, list):
                v = [v]

            df = df[df[f].isin(v)]

        # check length of dataframe
        if df.shape[0] == 0:
            raise ValueError(
                "'{0:}' did not return any results in field '{1:s}'".format(v, f)
            )

        # reset index if applicable
        if reset_index:
            # reset indexes
            df = df.reset_index(drop=True)

    return df


def create_dataframe_multinode(_list_df, timestep_seconds=None, verbose=False):
    if timestep_seconds is None:
        # Compute average dt
        _mean_dt = [(_df.index[-1]-_df.index[0]).total_seconds() / (len(_df)-1) for _df in _list_df]
        if verbose:
            print(f"Average dt per node: {_mean_dt}")
        dt = sum(_mean_dt) / len(_mean_dt)
    else:
        dt = timestep_seconds
    if verbose:
        print(f"Using dt = {dt} seconds")

    # Find common start time
    max_start_time = max([_df.index[0] for _df in _list_df])
    if verbose:
        print(f"Common start time = {max_start_time}")

    # Find common end time
    min_end_time = min([_df.index[-1] for _df in _list_df])
    if verbose:
        print(f"Common end time = {min_end_time}")

    # Create new index
    n = int((min_end_time - max_start_time).total_seconds() / dt) + 1
    new_idx = pd.Index([max_start_time + pd.to_timedelta(i * dt, unit='s') for i in range(n)])

    # Use new index
    _dfs = []
    for _df in _list_df:
        _idx = _df.index # old index
        _new_idx = new_idx.difference(_idx) # new index, removing duplicates

        # add new index to dataframe and interpolate values
        _df = pd.concat([_df, pd.DataFrame(index=_new_idx)]).sort_index()
        _df = _df.interpolate(method='polynomial', order=1)

        # remove old index
        _df = _df.drop(_idx.difference(new_idx))

        # store
        _dfs.append(_df)

    # Create single dataframe
    df = pd.DataFrame(index=new_idx)
    for _i, _df in enumerate(_dfs):
        _df_prefix = _df.add_prefix(f"{_i}_")
        df = df.add(_df_prefix, fill_value=0)

    return df


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
