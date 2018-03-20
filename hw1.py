import sys
import numpy as np
import pandas as pd

test_path = sys.argv[1]
predict_path = sys.argv[2]


def get_names(feature_selected, hours):
    names = []
    for feature in feature_selected:
        tmp = []
        for hour in hours:
            name = feature + '^' + str(hour)
            tmp.append(name)
        names += tmp[::-1]
    return names


selected_names = get_names(
    ['PM2.5', 'PM10', 'CO', 'SO2', 'O3', 'WIND_SPEED'], [1])
selected_names += ['BIAS']


# load testing data
test_csv = pd.read_csv(test_path, encoding='big5', names=[
                       'Id', 'Type', 'hour-9', 'hour-8', 'hour-7', 'hour-6', 'hour-5', 'hour-4', 'hour-3', 'hour-2', 'hour-1'])
test_raw = test_csv.drop(columns='Id')
test_raw.set_index('Type', inplace=True)
test_raw[test_raw == 'NR'] = 0
test_raw = test_raw.apply(pd.to_numeric, downcast='float')

feature_names = []
for feature in test_raw.index[:18]:
    for j in range(9):
        feature_names.append(feature + '^' + str(9 - j))

# testing data preprocessing
test_dataset = []
for i in range(260):
    start = i * 18
    datapd = test_raw.iloc[start: start + 18, :].copy()
    data = datapd.values.reshape(-1)
    test_dataset.append(data)
test_dataset = np.array(test_dataset)
test_organized = pd.DataFrame(data=test_dataset, columns=feature_names)
test_organized['BIAS'] = 1.0

test_organized[test_organized < 0] = -test_organized[test_organized < 0]
ts = test_organized[selected_names].values

# load model
weights = np.load('hw1_best.npy')

# make prediction
answer = np.dot(ts, weights)

# save prediction to output file
pd.DataFrame(data={'id': ['id_' + str(i) for i in range(len(answer))],
                   'value': answer}).to_csv(predict_path, index=False)
