import os
from toolz.curried import *
from util import *

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from pytorch_forecasting import TimeSeriesDataSet
from dataloader import subset_sampler
from models import RNN

year, month = 2021, 4
carpark_dir = "/home/xiucheng/data/carparking"
month_dir = os.path.join(carpark_dir, f"{year}/{month:02}")
carpark_meta = load_carpark_meta(os.path.join(carpark_dir, "carpark_meta.geojson"))

carpark_data_names = [
    "carpark_data_2021_04_16-20.json",
]

carpark_data = load_carpark_data_list(
    [os.path.join(carpark_dir, name) for name in carpark_data_names]
)

carpark_data = carpark_remove_outlier(carpark_data)
carpark_data = carpark_data.fillna(method="ffill")
carpark_data = carpark_add_date(carpark_data)
carpark_data = carpark_remove_constant(carpark_data)
carpark_data["min"] = (carpark_data.tid % 1440 // 5).astype(str)

device = torch.device("cuda:0")
batch_size = 200
context_length = 10*60
prediction_length = 2*60

training_cutoff = 3*24*60
validation_cutoff = 4*24*60

training = TimeSeriesDataSet(
    carpark_data[lambda x: x.tid <= training_cutoff],
    time_idx="tid",
    group_ids=["sid"],
    target="AvailableLots",
    time_varying_unknown_reals=["AvailableLots"],
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
    time_varying_known_categoricals=["min"],
)

validation = TimeSeriesDataSet.from_dataset(
    training,                                         
    carpark_data[lambda x: x.tid <= validation_cutoff],
    min_prediction_idx=training_cutoff + 1
)

test = TimeSeriesDataSet.from_dataset(
    training, 
    carpark_data,
    min_prediction_idx=validation_cutoff + 1
)

train_sampler = subset_sampler(training, 0.5, random=True)
val_sampler = subset_sampler(validation, 0.05)
test_sampler = subset_sampler(test, 0.05)

test_dataloader = test.to_dataloader(batch_size=500, num_workers=8,
                                     shuffle=False, sampler=test_sampler)
                                    
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, 
                                    patience=10, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=10,
    gpus=[0],
    weights_summary="top",
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=500
)
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = RNN.load_from_checkpoint(best_model_path).to(device)
print(best_model_path)
predictions = best_model.predict(test_dataloader, batch_size=500, show_progress_bar=True)