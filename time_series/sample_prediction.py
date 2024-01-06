import numpy as np
import pandas as pd
from resnet import Resnet
from data_preprocessing import load_raw_data

path = 'time_series/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate = 100
df = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')

df = df.iloc[53]

# Load raw signal data
df['ecg_data'] = load_raw_data(df['filename_lr'], sampling_rate, path)
X = np.expand_dims(df['ecg_data'], axis=0)
output_dir = 'time_series/model/'
resnet_model = Resnet(output_dir, input_shape=[1000, 12], n_classes=52)
prediction, _ = resnet_model.predict(X)
prediction = prediction[0]


with open('time_series/labels.json', 'r') as f:
    labels = f.read()

print("predicted", prediction)
masked_arr = (prediction > 0.5).astype(int)
print("Len masked", len(masked_arr))
result_list = [labels[i] for i in range(len(masked_arr)) if masked_arr[i] == 1]

print("Actual", df['scp_codes'])
print("Predicted", result_list)
