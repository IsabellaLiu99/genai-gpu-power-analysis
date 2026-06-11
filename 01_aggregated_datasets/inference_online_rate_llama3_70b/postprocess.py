import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import Union
from tqdm import tqdm
import sys
import json
sys.path.append(str(Path(__file__).parent.parent))
from utilities import *

DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"
PATH_DATA = Path(__file__).parent.parent.parent / '00_raw_datasets'
HERE = Path(__file__).parent

# device power limits in Watts: if values are higher, there was an error
DEVICE_LIMITS = {
    "CPU": 800,
    "GPU": 800,
}


def visual_check(series, val, val_diff, series_truncated, title=""):
    
    fig, axes = plt.subplots(3,1, figsize=(12,18),sharex=True)

    index = series.index

    # 1. original power and rolling average
    ax = axes[0]
    ax.plot(index, series.values, linewidth=1, label="original")
    ax.plot(index, val.values, linewidth=1, label="rolling")
    ax.legend()

    # 2. rate of change
    ax = axes[1]
    ax.plot(index, series.diff(), linewidth=1, label="original")
    ax.plot(index, val_diff, linewidth=1, label="rolling")
    ax.legend()

    # 3. truncated portions
    ax = axes[2]
    ax.plot(index, series.values, linewidth=1, label="original")
    ax.plot(series_truncated.index, series_truncated.values, linewidth=1, label="truncated")
    ax.legend()
    
    fig.suptitle(title)
    
    return fig, axes


def adjust_start_time(series, plot_flag=True, title="", raise_error=True):
    
    # approximate values
    idle_power = 700
    max_power = 3000

    # assume workload start is 50% increase in power
    threshold_max = 0.5*(max_power-idle_power)+ idle_power

    # assume noise threshold for rate of change in power is 1% of max swing in one timestep
    threshold_noise = 0.005*(max_power-idle_power)
    
    # calcualate rate of change of rolling average over 1 seconds (10 x 0.1s timesteps)
    val = series.rolling(10).mean().copy()
    val_diff = val.diff().copy()

    above_t = val[val> threshold_max]
        
    # first index
    try:
        idx_first_above_t = above_t.index[0]
    except Exception as e:
        fig, axes = visual_check(series, val, val_diff, series, title=title)
        return series, fig, axes, True
        
    
    # Walk index back to last time rate was less than noise threshold
    less = val_diff.loc[:idx_first_above_t]
    less = less[less < threshold_noise]
    idx_last_n = less.index[-1]
    
    if raise_error:
        if idx_last_n == series.index[0]:
            raise Exception("Filtering did not work")

    series_truncated = series.loc[idx_last_n:]
    
    if plot_flag:
        fig, axes = visual_check(series, val, val_diff, series_truncated, title=title)
    else:
        fig = None
        axes = None
    
    return series_truncated, fig, axes, False


def load_jsonl(filepath):
    """
    Load a JSONL file where each line is a separate JSON object.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the JSONL file
    
    Returns:
    --------
    list of dict
        List of dictionaries, one per line in the file
    """
    filepath = Path(filepath)
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    return data
    

def extract_series(record_metadata:dict, df_nvml_in:pd.DataFrame, df_rapl_in: pd.DataFrame, flag_check=False)->pd.DataFrame:
    """
    Extracts and aligns GPU and CPU power timeseries for a single inference record.

    Parameters
    ----------
    record_metadata : dict
        Metadata for the inference record, containing 'start_time' and 'end_time' as strings.
    df_nvml_in : pd.DataFrame
        DataFrame containing GPU power readings, indexed by timestamp.
    df_rapl_in : pd.DataFrame
        DataFrame containing CPU power readings, indexed by timestamp.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by elapsed seconds ('timestep[s]'), containing resampled and aligned GPU and CPU power columns,
        as well as total GPU, CPU, and combined power.
    """
    
    df_nvml = df_nvml_in.copy(deep=True)
    df_rapl = df_rapl_in.copy(deep=True)
        
    key_start = "start_time"
    key_end = "end_time"
        
    # extract timeseries
    start_time = pd.to_datetime(record_metadata[key_start])
    end_time = pd.to_datetime(record_metadata[key_end])

    # index
    series_nvml = df_nvml.loc[start_time:end_time]
    series_rapl = df_rapl.loc[start_time:end_time]

    # resample
    df_record = create_dataframe_multinode([series_nvml, series_rapl], 0.001)

    # keep only power columns and rescale to Watts
    # TODO: hardcoding column names is not robust (e.g. more/less than one node might be used)
    cols_power_nvml = ['0_gpu-0[mW]', '0_gpu-1[mW]', '0_gpu-2[mW]', '0_gpu-3[mW]']
    cols_power_nvml_new = [f"{c.split('[')[0]}[W]" for c in cols_power_nvml]
    for c_old, c_new in zip(cols_power_nvml, cols_power_nvml_new):
        df_record[c_new] = df_record[c_old]*1e-3
        
    cols_power_rapl = ['1_cpu-0[W]', '1_cpu-0-core[W]', '1_cpu-1[W]', '1_cpu-1-core[W]']

    # merge and clean
    series = df_record[cols_power_nvml_new +cols_power_rapl].copy()
    series = _clean_frame(series, "CPU")
    series = _clean_frame(series, "GPU")
    
    # calculate totals
    series["total_GPU[W]"] = series[cols_power_nvml_new].sum(axis=1)
    series["total_CPU[W]"] = series[cols_power_rapl].sum(axis=1)
    series["power[W]"] = series[cols_power_nvml_new+cols_power_rapl].sum(axis=1) 

    # re-index
    series["timestep[s]"] = (series.index-series.index[0]).total_seconds()
    series = series.reset_index(drop=True).set_index("timestep[s]")
    
    # adjust start time
    # series, fig, axes, failed = adjust_start_time(series["power[W]"], plot_flag=flag_check, raise_error=True)
    if flag_check:
        fig, axes = visual_check(series["power[W]"], series["power[W]"], series["power[W]"], series["power[W]"])
    else:
        fig, axes = None, None
    series = series["power[W]"]
    
    # re-index
    series = series.to_frame()
    series["timestep[s]"] = (series.index-series.index[0])
    series = series.reset_index(drop=True).set_index("timestep[s]")
    
    # calculate true elapsed time
    elapsed_seconds = series.index[-1]
    
    return series, elapsed_seconds, fig, axes, False


def read_power_log(path: Union[Path, str]) -> pd.DataFrame:
    """
    Reads a power log file, skipping commented lines, and returns a pandas DataFrame
    indexed by timestamp.

    Args:
        path (Union[Path, str]): Path to the power log file.

    Returns:
        pd.DataFrame: DataFrame containing the power log data indexed by timestamp.
    """
    path = Path(path)

    # Read the file, skipping commented lines
    df = pd.read_csv(
        path,
        delim_whitespace=True,  # split on whitespace
        skiprows=2
    )

    cols = df.columns[1:].tolist()
    df = df.iloc[:, :-1]
    df.columns = cols

    df['timestamp'] = pd.to_datetime(df["timestamp"], format=DATETIME_FORMAT)
    df = df.set_index("timestamp")

    return df

def _clean_frame(df, device_name):
    for col in df.columns:
        if device_name in col.upper():
            df.loc[
                df[col] > DEVICE_LIMITS[device_name], col
            ] = None
            df[col] = df[col].bfill()
    return df



def main(output_directory, results_directory):
    
    output_directory = Path(output_directory)
    results_directory = Path(results_directory)
    
    # directories
    output_directory.mkdir(parents=True, exist_ok=True)
    plots_directory = output_directory / "plots_check"
    
    # load metadata
    config = load_json(results_directory / "config.json")
    
    # read results
    list_results = load_json(results_directory / "log.json")
    df_results = pd.DataFrame(list_results)
    
    # convert to datetimes
    df_results = df_results.rename(columns={"start_time":"start_time_exc","end_time":"end_time_exc"})
    cols_timestamp = ['start_time_exc', 'end_time_exc']
    for col in cols_timestamp:
        df_results[col] = pd.to_datetime(df_results[col], format=DATETIME_FORMAT)
        
    # remove 10 seconds from boths sides
    df_results["start_time_exc"] = df_results["start_time_exc"] +pd.Timedelta(1, "minutes")
    end_time_new = (df_results["start_time_exc"] +pd.Timedelta(3, "minutes")).clip(upper=df_results["end_time_exc"])
    df_results["end_time_exc"] = end_time_new
    
    # read vllm log
    if os.path.exists(results_directory / "log_vllm.json"):
        log_vllm = load_jsonl(results_directory / "log_vllm.json")
        
        # convert to dataframe and merge
        df_log_vllm = pd.DataFrame(log_vllm)
        
        df_log_vllm["start_time_vllm"] = pd.to_datetime(df_log_vllm["date"], format="%Y%m%d-%H%M%S")
        df_log_vllm["end_time_vllm"] = df_log_vllm["start_time_vllm"] + pd.to_timedelta(df_log_vllm["duration"], "seconds")
        
        # merge
        df_results = pd.merge(
            df_results,
            df_log_vllm,
            how="outer",
            on="id",
            validate="1:1"
        )
        
    else:
        pass
    
    # convert to datetime
    list_results = df_results.to_dict(orient="records")
    
    nodelist = config['slurm']['SLURM_NODELIST']
    
    # Get all file names in results_directory that contain both nodelist and "nvml_wattameter_"
    nvml_file = [
        f for f in os.listdir(results_directory)
        if "nvml_wattameter_" in f and str(nodelist) in f
    ]
    rapl_file = [
        f for f in os.listdir(results_directory)
        if "rapl_wattameter_" in f and str(nodelist) in f
    ]
    
    assert len(nvml_file) == 1, f"Expected one NVML file, found {len(nvml_file)}: {nvml_file}"
    assert len(rapl_file) == 1, f"Expected one RAPL file, found {len(rapl_file)}: {rapl_file}"
    nvml_file = nvml_file[0]
    rapl_file = rapl_file[0]
        
    # load power logs
    df_nvml = read_power_log(path=results_directory / nvml_file)
    df_rapl = read_power_log(path=results_directory / rapl_file)
    
    # for each record, extract power profile
    runs_directories = output_directory
    runs_directories.mkdir(exist_ok=True, parents=True)
    
    # list_results = list_results[:5]
    list_results = tqdm(list_results, desc="Processing records")
    
    # spot check a few
    flag_check_vec = [False] * len(list_results)
    for idx in random.sample(range(len(list_results)), 10):
        flag_check_vec[idx] = True
    
    for irecord, record in enumerate(list_results):
        
        flag_check = flag_check_vec[irecord]
            
        # use execution
        key_start = "start_time_exc"
        key_end = "end_time_exc"
        
        record["start_time"] = record[key_start]
        record["end_time"] = record[key_end]
        
        # extract series
        save_path = runs_directories / f"{irecord:06d}.parquet"
        series_record, elapsed_seconds, fig, axes, failed = extract_series(record, df_nvml, df_rapl, flag_check)
        series_record = series_record[["power[W]"]]
        series_record.to_parquet(save_path)
        
        # save visual check            
        if failed:
            record["run_completed"] = False
        else:
            record["run_completed"] = True
              
        # add to metadata
        record["execution_time_seconds"] = elapsed_seconds
        record["path_run"] = save_path
        record["peak_power[W]"] = series_record["power[W]"].max()
        record["mean_power[W]"] = series_record["power[W]"].mean()
        
    # save metadata
    df_list_results = pd.DataFrame(list_results)
    df_list_results.to_csv(HERE / "metadata.csv", index=False)
    
    return df_list_results
    
    
    
if __name__ == "__main__":
    
    results_directory = PATH_DATA / "inference_online_rate_llama3_70b"
    output_directory = HERE / "results"
    
    main(output_directory, results_directory)
        



