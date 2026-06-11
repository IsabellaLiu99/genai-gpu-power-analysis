import pandas as pd
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utilities import *

DATETIME_FORMAT = "%Y-%m-%d_%H:%M:%S.%f"
PATH_DATA = Path(__file__).parent.parent.parent / '00_raw_datasets'


def _read_rapl_log(path):
    path = Path(path)

    df = pd.read_csv(
        path,
        sep=r'\s+',
        comment='#',
        header=None
    )
    # Optionally assign column names
    
    
    columns = [
        "timestamp",
        "reading-time[ns]",
        "cpu-0[uJ]",
        "cpu-0-core[uJ]",
        "cpu-1[uJ]",
        "cpu-1-core[uJ]",
        "cpu-0[W]",
        "cpu-0-core[W]",
        "cpu-1[W]",
        "cpu-1-core[W]"
    ]
    
    if len(df.columns) != len(columns):
        raise Exception("Incorrect column length")
    
    df.columns = columns
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=DATETIME_FORMAT)
    df = df.set_index("timestamp")
    
    # aggregate power colums
    columns_power = [
        "cpu-0[W]",
        "cpu-0-core[W]",
        "cpu-1[W]",
        "cpu-1-core[W]"
    ]
    
    df["power_W"] = df[columns_power].sum(axis=1, numeric_only=True)
        
    return df["power_W"]


def _read_nvml_log(path):
    path = Path(path)

    df = pd.read_csv(
        path, sep=r'\s+', 
        comment='#', 
        header=None
    )
    
    with open(path) as f:
        header = f.readline().replace('# ', '').strip().split()
    df.columns = header
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=DATETIME_FORMAT)
    df = df.set_index("timestamp")
    
    # aggregate power colums
    columns_power = [
        "gpu-0[mW]", 
        "gpu-1[mW]",
        "gpu-2[mW]",
        "gpu-3[mW]"
    ]
    
    df["power_W"] = (df[columns_power]*1e-3).sum(axis=1, numeric_only=True)
        
    return df["power_W"]


def extract_job_record(df_job):
        
    map_device = {
        "rapl":"CPU",
        "nvml":"GPU"
    }

    CPU_counter = -1
    GPU_counter = -1
    res = []
    for record in df_job.to_dict(orient="records"):

        if record["devices"] == "rapl":
            series = _read_rapl_log(record["path"])
            CPU_counter += 1
            counter = CPU_counter
        elif record["devices"] == "nvml":
            series = _read_nvml_log(record["path"])
            GPU_counter += 1
            counter = GPU_counter
        else:
            raise KeyError(f"Unrecognized device: {record["devices"]}")

        series.name = f"{map_device[record["devices"]]}_{counter}"

        res.append(series)
        
    return res


def resample_series(list_df_record, timestep_seconds=None):
    
    # resample to same index
    df_record = create_dataframe_multinode(list_df_record, timestep_seconds)
    
    # get total power
    total_power = df_record.sum(axis=1, numeric_only=True)
    total_power.name = "power[W]"
    total_power.index.name = "timestamp"

    total_power = total_power.to_frame()

    # re-index
    series = total_power
    series["timestep[s]"] = (series.index-series.index[0]).total_seconds()
    series = series.reset_index(drop=True).set_index("timestep[s]")
    
    return series


def get_job_series(df_job):
    
    # extract dataframe power record
    df_record = extract_job_record(df_job)
    
    # resample for total power consumption
    df_series = resample_series(df_record, 0.2)
    
    # # resample to 1Hz use backfill
    # index = df_series.index
    # new_index = np.round(np.arange(index[0], index[-1]+0.1, 0.1), decimals=1)
    # df_series = df_series.reindex(new_index).bfill()
    
    return df_series


def main(path_data, data_map, output_directory):
    
    list_metadata = []
    counter_save = 0
    for model, model_runs in data_map.items():
        for nodes_key, nodes_name in model_runs["nodes"].items():
            path = path_data / model_runs["path"] / nodes_name
            
            # get all folders
            list_files = os.listdir(path)
            
            # parse folders into list of dictionaries
            res = []
            for tfile in list_files:
                list_parts = (tfile.split(".")[0]).split("_")
                res.append({
                    "devices": list_parts[0],
                    "node": list_parts[7],
                    "path": path / tfile, 
                    "slurmid":list_parts[5]
                })
            df_res = pd.DataFrame(res)
            
            # unique ids
            list_slurmid = sorted(df_res["slurmid"].unique())
                        
            # extract timeseries for each unique job
            for ijob, job in enumerate(list_slurmid):
                df_job = filter_df(df_res, ["slurmid"], [job])
                
                # validate shape
                if df_job.shape[0] != 2*nodes_key:
                    print(f"WARNING! incorrect number of records:\n   expected:{nodes_key*2}\n   measured:{df_job.shape[0]}")
                    
                # extract timeseries
                path_save = output_directory / f"{counter_save:06d}.parquet"
                timeseries = get_job_series(df_job)
                timeseries.to_parquet(path_save)
                
                metadata = {
                    "model": model,
                    "nodes": nodes_key,
                    "repeat": ijob,
                    "path_save": Path(*path_save.parts[-3:]),
                    "slurmid": job
                }            

                # get last 
                
                list_metadata.append(metadata)
                counter_save += 1
                print(path_save)
                
    return pd.DataFrame(list_metadata)
    


if __name__ == "__main__":
    
    HERE = Path(__file__).parent
    
    output_directory = HERE / "results"
    os.makedirs(output_directory, exist_ok=True)
    
    data_map = {
        "llama2_70b_lora": {
            "path": "training_llama2_70b_lora",
            "nodes": {
                2: "2node",
                4: "4node",
                8: "8node",
                16: "16node"
            }
        },
        "stable_diffusion": {
            "path": "training_stable_diffusion",
            "nodes": {
                2: "2node",
                4: "4node",
                8: "8node",
                16: "16node"
            }
        }
    }
        
    # run
    df_metadata = main(PATH_DATA, data_map, output_directory)
    
    # save
    df_metadata.to_csv(HERE / "metadata.csv")