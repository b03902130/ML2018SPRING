import sys
import pandas as pd
import numpy as np

train_path = sys.argv[1]


def get_names(feature_selected, hours):
    names = []
    for feature in feature_selected:
        tmp = []
        for hour in hours:
            name = feature + '^' + str(hour)
            tmp.append(name)
        names += tmp[::-1]
    return names


# load data
train_csv = pd.read_csv(train_path, encoding='big5')
train_raw = train_csv.iloc[:, 2:].rename(columns={'測項': 'type\hour'})
train_raw.set_index('type\hour', inplace=True)
train_raw[train_raw == 'NR'] = 0
train_raw = train_raw.apply(pd.to_numeric, downcast='float')

# training data preprocessing
i = 0
hours = []
while i < train_raw.shape[0]:
    tmp = train_raw.iloc[i:i + 18, :]
    hours.append(tmp)
    i += 18
train_slidewindow = pd.concat(hours, axis=1)
train_slidewindow.columns = range(5760)

feature_names = []
for feature in train_slidewindow.index:
    for j in range(9):
        feature_names.append(feature + '^' + str(9 - j))

pm2_index = 9
slidewindow = train_slidewindow.values
train_dataset = []
for month in range(12):
    for start_hour in range(471):
        index = month * 480 + start_hour
        data = slidewindow[:, index: index + 9].reshape(-1)
        answer = slidewindow[pm2_index, index + 9]
        data = np.append(data, [1, answer])
        train_dataset.append(data)
train_dataset = np.array(train_dataset, dtype=np.float32)
train_organized = pd.DataFrame(
    data=train_dataset, columns=feature_names + ['BIAS', 'PM2.5'])

# drop PM2.5 <= 0, fill other negative values with positive
data = train_organized
data_dropped = data.drop(data[data['PM2.5'] <= 0].index)
data_dropped[data_dropped < 0] = -data_dropped[data_dropped < 0]

# select features
selected_names = get_names(
    ['PM2.5', 'PM10', 'CO', 'SO2', 'O3', 'WIND_SPEED'], [1])
selected_names += ['BIAS']

xs = data_dropped[selected_names].values
ys = data_dropped['PM2.5'].values


def gradient_descient_fit(Xtrain, Ytrain, loop_count, learning_rate):
    feature_num = Xtrain.shape[1]
    parameters = np.zeros(feature_num, dtype=np.float32)
    gradient_root = np.zeros(feature_num, dtype=np.float32)

    for loop in range(loop_count):
        # CALCULATE LOSS
        Ypredict = np.dot(Xtrain, parameters)
        Ydiff = Ytrain - Ypredict
        loss = np.sqrt((Ydiff ** 2).sum() / len(Ydiff))

        # CALCULATE GRADIENT
        gradient = np.dot(-2 * Ydiff, Xtrain)
        gradient_root = np.sqrt(gradient_root ** 2 + gradient ** 2)

        # UPDATE PARAMETERS
        parameters -= learning_rate * gradient / gradient_root

        # if loop % 1000 == 0:
        #     print('loss: ' + str(loss))

    return parameters


# train model
weights = gradient_descient_fit(xs, ys, 100000, 0.1)

# save model
np.save('hw1.npy', weights)
