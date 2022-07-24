
import numpy as np
import pandas as pd
import geopandas as gpd
import json, os


from datetime import datetime
from tqdm import tqdm
from pandas import DataFrame
from geopandas import GeoDataFrame
from shapely.geometry import Point
from typing import List
from toolz.curried import *

def read_carpark_json_hour(json_file: str) -> DataFrame:
    with open(json_file, "r") as f:
        data = json.load(f)["data"]
    carpark = pd.DataFrame(data)
    carpark.datetime = carpark.datetime.map(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S+08:00"))
    carpark.datetime = carpark.datetime.dt.floor("T") #perform floor operation to the minute
    carpark.Location = carpark.Location.map(
        lambda x : pipe(x.split(),
                        map(float), 
                        tuple,
                        lambda x: Point(x[::-1]))
    )
    return carpark

def read_carpark_json_day(daydir: str) -> DataFrame:
    carparks = []
    for file in tqdm(os.listdir(daydir)):
        carparks.append(read_carpark_json_hour(os.path.join(daydir, file)))
    return pd.concat(carparks).reset_index(drop=True)

def read_carpark_json_days(monthdir: str, days: List[int]) -> DataFrame:
    carparks = []
    for day in days:
        daydir = os.path.join(monthdir, f"{day:02}")
        if os.path.exists(daydir):
            carparks.append(read_carpark_json_day(daydir))
    return pd.concat(carparks).reset_index(drop=True)

def read_carpark_json_month(monthdir: str) -> DataFrame:
    carparks = []
    for day in os.listdir(monthdir):
        daydir = os.path.join(monthdir, day)
        carparks.append(read_carpark_json_day(daydir))
    return pd.concat(carparks).reset_index(drop=True)

def save_carpark_meta(carpark: DataFrame, carpark_meta_path: str) -> GeoDataFrame:
    """
        carpark_meta = gpd.read_file("carpark_meta.geojson")
    """
    meta_cols = ["CarParkID", "Location", "Area", "Development", "LotType", "Agency"]
    carpark_meta = carpark[meta_cols]\
                    .drop_duplicates(subset="CarParkID", ignore_index=True)
    carpark_meta = gpd.GeoDataFrame(carpark_meta, geometry="Location")

    mapping = {carparkid: i for (i, carparkid) in enumerate(carpark_meta["CarParkID"].unique())}
    carpark_meta["sid"] = carpark_meta["CarParkID"]
    carpark_meta = carpark_meta.replace({"sid": mapping})

    carpark_meta.to_file(carpark_meta_path, driver="GeoJSON")
    print(f"{carpark_meta.shape[0]} spatial objects have been saved.")
    
    return carpark_meta

def save_carpark_data(carpark: DataFrame, carpark_meta: GeoDataFrame, carpark_data_path: str) -> None:
    """
        carpark_data = pd.read_json("carpark_data.json")
    """
    
    data_cols = ["CarParkID", "datetime", "AvailableLots"]
    carpark_data = carpark.loc[:, data_cols]
    carpark_data = carpark_data[carpark_data.CarParkID.isin(carpark_meta.CarParkID)]

    ## mapping carparkid from str to int, replace is slow
    #carpark_data.rename(columns={"CarParkID": "sid"}, inplace=True)
    #carpark_data.replace({"sid": mapping}, inplace=True)
    carpark_data = pd.merge(carpark_data, carpark_meta[["CarParkID", "sid"]],
        left_on="CarParkID", right_on="CarParkID", how="left").drop("CarParkID", axis=1)

    carpark_data.to_json(carpark_data_path)
    print(f"{carpark_data.shape[0]} rows have been saved.")

def load_carpark_meta(carpark_meta_path: str) -> GeoDataFrame:
    carpark_meta = gpd.read_file(carpark_meta_path)
    return carpark_meta

def carpark_add_date(carpark_data: DataFrame) -> DataFrame:
    carpark_data = carpark_data.reset_index()
    carpark_data["date"] = carpark_data["tid"]
    carpark_data["tid"] = ((carpark_data["tid"] - carpark_data["tid"].min()) 
                            / np.timedelta64(1, "m")).astype(np.int32)
    return carpark_data

def load_carpark_data(carpark_data_path: str, reindex=False) -> DataFrame:
    """
    Usage:
    carpark_data = load_carpark_data(os.path.join(carpark_dir, "carpark_data_2021_04_16-20.json")
    """
    print(f"Loading {carpark_data_path}")
    carpark_data = pd.read_json(carpark_data_path)
    carpark_data.rename(columns={"datetime": "tid"}, inplace=True)

    carpark_data = carpark_data.groupby(["sid", "tid"]).agg(AvailableLots=("AvailableLots", "mean"))
    
    if reindex:
        sids = carpark_data.reset_index().sid.unique()
        tids = carpark_data.reset_index().tid.unique()
        tids = pd.date_range(start=tids.min(), end=tids.max(), freq="min")
        #tids = ((tids - tids.min()) / np.timedelta64(1, "m")).astype(np.int32)
        index = pd.MultiIndex.from_product([sids, tids], names=["sid", "tid"])
        carpark_data = carpark_data.reindex(index)

    return carpark_data

def load_carpark_data_list(carpark_data_paths: List[str]) -> DataFrame:
    """
    Usage:
    carpark_data_names = [
        "carpark_data_2021_04_16-20.json",
        "carpark_data_2021_04_21-25.json"
    ]
    carpark_data = load_carpark_data_list(
        [os.path.join(carpark_dir, name) for name in carpark_data_names]
    )    
    """
    carpark_data = pd.concat([
        load_carpark_data(carpark_data_path) for carpark_data_path in carpark_data_paths
    ])
    sids = carpark_data.reset_index().sid.unique()
    tids = carpark_data.reset_index().tid.unique()
    tids = pd.date_range(start=tids.min(), end=tids.max(), freq="min")
    index = pd.MultiIndex.from_product([sids, tids], names=["sid", "tid"])
    return carpark_data.reindex(index)

def carpark_remove_outlier(carpark_data: DataFrame) -> DataFrame:
    
    def move_average(x, n):
        w = np.ones(n) / n
        x_f = np.convolve(x, w, mode="valid")
        x_b = np.convolve(x[-2*n+2:][::-1], w, mode="valid")[::-1]
        return np.concatenate([x_f, x_b])

    sids = carpark_data.reset_index().sid.unique()
    for sid in tqdm(sids):
        x = carpark_data.loc[sid].fillna(method="ffill").values.ravel()
        m = move_average(x, 5)
        idx, = np.where(np.abs(x - m) > 50)
        carpark_data.loc[sid].iloc[idx] = np.nan

    return carpark_data

def carpark_remove_constant(carpark_data: DataFrame) -> DataFrame:
    nonconstant = carpark_data.groupby(["sid"]).apply(lambda x: x["AvailableLots"].std()) > 1.0
    nonconstant_sids = nonconstant[nonconstant].index
    return carpark_data[carpark_data.sid.isin(nonconstant_sids)]

# from gluonts.dataset.common import ListDataset
# def carpark_list2glonts_data(
#     carpark_data_list: List[DataFrame], context_length: int, 
#     prediction_length: int, prediction_interval: int
# ) -> Tuple[ListDataset, ListDataset]:
#     """
#     sids = carpark_data.reset_index().sid.unique()
#     carpark_data_list = [carpark_data.loc[sid].fillna(method="ffill") for sid in tqdm(sids)]
#     """

#     def to_gluonts(tids, vals):
#         vals = np.nan_to_num(vals)
#         return {"start": tids[0], "target": vals}
    
#     def to_gluonts_trn(one_carpark):
#         tids = one_carpark.index
#         vals = one_carpark.values.ravel()
#         #return [to_gluonts(tids[0:-prediction_interval], vals[0:-prediction_interval])]
#         return [to_gluonts(tids[0:2*24*60], vals[0:2*24*60])]
    
    
#     def to_gluonts_tst(one_carpark):
#         """
#         Produce `prediction_interval // prediction_length` test instances from `[-prediction_interval:]`.
#         """
#         tids = one_carpark.index
#         vals = one_carpark.values.ravel()
        
#         i = context_length+prediction_length
#         return [to_gluonts(tids[-i:], vals[-i:])] + [
#             to_gluonts(tids[-i-k:-k], vals[-i-k:-k])
#             for k in range(prediction_length, prediction_interval-prediction_length+1, prediction_length)
#         ]
    
#     # sids = carpark_data.reset_index().sid.unique()
#     # carpark_data_list = [carpark_data.loc[sid] for sid in tqdm(sids)]
#     trns = list(mapcat(to_gluonts_trn, tqdm(carpark_data_list)))
#     tsts = list(mapcat(to_gluonts_tst, tqdm(carpark_data_list)))
    
#     return ListDataset(trns, freq="1min"), ListDataset(tsts, freq="1min")