Code with reference from https://github.com/chaoshangcs/GTS

Train GTS model with Nottingham dataset 

Section 1: Prepare dataset 
1. Navigate to data/sensor_graph and run python findDistance_Nottingham.py 
2. Navigate to data and run python getAvailData_Nottingham.py
3. Navigate to base directory of this repository and run python -m scripts.generate_training_data --output_dir=data/Nottingham --traffic_df_filename=data/Nottingham_timestamp_availability_data.csv

Section 2: Train model 
1. In base directory of this repository, run python train.py --config_filename=data/model/Nottingham.yaml --temperature=0.5

Train GTS model with San Francisco (SFParks) dataset

Section 1: Prepare dataset
1. Navigate to data/sensor_graph and run python generateMidPoint_SFPark.py
2. Navigate to data/sensor_graph and run python findDistance_SFPark.py
3. Navigate to data and run python getAvailData_SFPark.py
4. Navigate to data and run python getCapacityData_SFPark.py
3. Navigate to base directory of this repository and run python -m scripts.generate_training_data_SFPark --output_dir=data/SFPark --traffic_df_filename=data/sfpark_timestamp_availability_data.csv

Section 2: Train model 
1. In base directory of this repository, run python train.py --config_filename=data/model/sfpark.yaml --temperature=0.5
