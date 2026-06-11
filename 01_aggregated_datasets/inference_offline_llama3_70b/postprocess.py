from pathlib import Path
import pandas as pd
from typing import Union
from tqdm import tqdm
import sys
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

def extract_series(record_metadata:dict, df_nvml_in:pd.DataFrame, df_rapl_in: pd.DataFrame)->pd.DataFrame:
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
     
    # extract timeseries
    start_time = pd.to_datetime(record_metadata["start_time"])
    end_time = pd.to_datetime(record_metadata["end_time"])

    # index
    series_nvml = df_nvml.loc[start_time:end_time]
    series_rapl = df_rapl.loc[start_time:end_time]

    # resample
    df_record = create_dataframe_multinode([series_nvml, series_rapl], 0.1)

    # keep only power columns and rescale to Watts
    # TODO: hardcoding column names is not robust (e.g. more/less than one node might be used)
    cols_power_nvml = ['0_gpu-0[mW]', '0_gpu-1[mW]', '0_gpu-2[mW]', '0_gpu-3[mW]']
    cols_power_nvml_new = [f"{c.split("[")[0]}[W]" for c in cols_power_nvml]
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
    
    return series


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
    
    # load configuration
    config = load_json(results_directory / "config.json")
    
    # read results
    list_results = load_json(results_directory / "log.json")
    df_results = pd.DataFrame(list_results)
    cols_timestamp = ['start_time', 'end_time']
    for col in cols_timestamp:
        df_results[col] = pd.to_datetime(df_results[col], format=DATETIME_FORMAT)
    list_results = df_results.to_dict(orient="records")
        
    # load power logs
    df_nvml = read_power_log(path=results_directory / f"nvml_wattameter_{config["SLURM_NODELIST"]}.log")
    df_rapl = read_power_log(path=results_directory / f"rapl_wattameter_{config["SLURM_NODELIST"]}.log")
    
    # for each record, extract power profile
    runs_directories = output_directory
    runs_directories.mkdir(exist_ok=True, parents=True)
    
    # list_results = list_results[:5]
    list_results = tqdm(list_results, desc="Processing records")
    for irecord, record in enumerate(list_results):
        
        # extract series
        save_path = runs_directories / f"{irecord:06d}.parquet"
        series_record = extract_series(record, df_nvml, df_rapl)
        series_record = series_record[["power[W]"]]
        series_record.to_parquet(save_path)
        
        # add to metadata
        record["path_run"] = save_path
        record["peak_power[W]"] = series_record["power[W]"].max()
        record["mean_power[W]"] = series_record["power[W]"].mean()
        
    # save metadata
    df_list_results = pd.DataFrame(list_results)
    df_list_results.to_csv(HERE / "metadata.csv", index=False)
    
    
if __name__ == "__main__":
    
    results_directory = PATH_DATA / "inference_offline_llama3_70b"
    output_directory = HERE / "results"
    
    main(output_directory, results_directory)



