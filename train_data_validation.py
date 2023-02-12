import numpy as np

PREPARED_DATA_DIR = 'prepared_data'
data_file = 'MVID_BBG004S68CP5_1m.csv'

prepared_data_dir = f"{PREPARED_DATA_DIR}/{data_file}"

f = "30_360"

TRAIN_TS = np.load(f"{prepared_data_dir}/{f}/train_ts.npy")
TRAIN_VALUE = np.load(f"{prepared_data_dir}/{f}/train_value.npy")
VALIDATE_TS = np.load(f"{prepared_data_dir}/{f}/validate_ts.npy")
VALIDATE_VALUE = np.load(f"{prepared_data_dir}/{f}/validate_value.npy")

print(len(TRAIN_TS))
print(len(TRAIN_VALUE))
print(len([val for val in TRAIN_VALUE if (val == [1, 0, 0]).all()]))
print(len([val for val in TRAIN_VALUE if (val == [0, 1, 0]).all()]))
print(len([val for val in TRAIN_VALUE if (val == [0, 0, 1]).all()]))
print(len([val for val in TRAIN_VALUE if (val == [0, 0, 0]).all()]))
