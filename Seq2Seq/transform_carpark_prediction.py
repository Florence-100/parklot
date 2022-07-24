import os, time, argparse
import util

parser = argparse.ArgumentParser(description="info")
parser.add_argument("--year", type=int, default=2021)
parser.add_argument("--month", type=int, default=6)
parser.add_argument("--startday", type=int, default=16)
parser.add_argument("--endday", type=int, default=17)
parser.add_argument("--duration", type=int, default=2)
parser.add_argument("--src_dir", type=str, default="/home_nfs/jiangyue/s3drive")
parser.add_argument("--dst_dir", type=str, default="/home_nfs/jiangyue/data/carparking")
args = parser.parse_args()

year, month = args.year, args.month
startday, endday, duration = args.startday, args.endday, args.duration

src_dir = args.src_dir
dst_dir = args.dst_dir
month_dir = os.path.join(src_dir, f"{year}/{month:02}")
print(f"Reading from {month_dir}...")
## constructing carpark_meta
carpark = util.read_carpark_json_days(month_dir, range(startday, startday+duration))
carpark_meta_path = os.path.join(dst_dir, "carpark_meta_present.geojson")
if not os.path.exists(carpark_meta_path):
    carpark_meta = util.save_carpark_meta(carpark, carpark_meta_path)
else:
    carpark_meta = util.load_carpark_meta(carpark_meta_path)
## constructing carpark_meta 

## saving carpark_data
tic = time.time()
carpark_data_name = f"carpark_data_{year}_{month:02}_{startday:02}-{min(endday, startday+duration-1):02}.json"
print(f"To save {carpark_data_name}")
util.save_carpark_data(carpark, carpark_meta, os.path.join(dst_dir, carpark_data_name))
print(f"save_carpark_data took {time.time() - tic} seconds.")

for day in range(startday+duration, endday, duration):
    carpark = None # release the memory of carpark (large object)
    carpark = util.read_carpark_json_days(month_dir, range(day, day+duration))
    carpark_data_name = f"carpark_data_{year}_{month:02}_{day:02}-{min(endday, day+duration-1):02}.json"
    print(f"To save {carpark_data_name}")
    util.save_carpark_data(carpark, carpark_meta, os.path.join(dst_dir, carpark_data_name))
