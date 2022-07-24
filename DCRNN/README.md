Code with reference from https://github.com/chnsh/DCRNN_PyTorch

Train DCRNN model with Nottingham dataset 

Section 1: Prepare dataset 
1. Navigate to data/sensor_graph and run python findDistance_Nottingham.py 
2. Navigate to data and run python getAvailData_Nottingham.py
3. Navigate to base directory of this repository and run python -m scripts.generate_training_data --output_dir=data/Nottingham --traffic_df_filename=data/Nottingham_timestamp_availability_data.csv
4. In base direcotry of this repository, run python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/Nottingham_carparkIds.txt --normalized_k=0.1 --output_pkl_filename=data/sensor_graph/adj_mat_nottingham.pkl

Section 2: Train model 
1. In base directory of this repository, run python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_nottingham.yaml

Train DCRNN model with San Francisco (SFParks) dataset

Section 1: Prepare dataset
1. Navigate to data/sensor_graph and run python generateMidPoint_SFPark.py
2. Navigate to data/sensor_graph and run python findDistance_SFPark.py
3. Navigate to data and run python getAvailData_SFPark.py
4. Navigate to data and run python getCapacityData_SFPark.py
3. Navigate to base directory of this repository and run python -m scripts.generate_training_data_SFPark --output_dir=data/SFPark --traffic_df_filename=data/sfpark_timestamp_availability_data.csv
4. In base direcotry of this repository, run python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/sfpark_segmentIds.txt --normalized_k=0.1 --output_pkl_filename=data/sensor_graph/adj_mat_sfpark.pkl

Section 2: Train model 
1. In base directory of this repository, run python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_sfpark.yaml
