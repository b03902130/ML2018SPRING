import sys
import copy
import pandas as pd
import numpy as np

train_path = sys.argv[1]    # for train.csv
trainX_path = sys.argv[2]   # for train_X
trainY_path = sys.argv[3]   # for train_Y


def standardize(data, scaler=0):
    if scaler == 0:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        scaler = (mean, std)
        return (data - mean) / std, scaler
    else:
        mean, std = scaler
        return (data - mean) / std


def accuracy_score(true, predict):
    return (true == predict).sum() / len(true)


def gaussian(x, mean=50, std=20):
    exp = -((x - mean) ** 2) / (2 * (std ** 2))
    return (1 / np.sqrt(2 * np.pi * std)) * (np.e ** exp)


raw_train = pd.read_csv(train_path)
raw_trainX = raw_train.iloc[:, :-1]
raw_trainY = raw_train.iloc[:, -1]


raw_X_train = pd.read_csv(trainX_path)
raw_Y_train = pd.read_csv(trainY_path, header=None).iloc[:, 0]


def train_preprocessing(data):
    tmp = data.copy(deep=True)
    tmp['gain2'] = tmp['capital_gain'] ** 2
    tmp['diff'] = tmp['capital_gain'] - tmp['capital_loss']
    tmp['new_age'] = np.absolute(1 / (tmp['age'] - 50.01))
    tmp['hpw'] = (tmp['hours_per_week'] > 39).apply(lambda x: 1 if x else 0)
    tmp['edu_num'] = (raw_trainX.iloc[train_id].education_num.values) ** 2
    
#     tmp['age_label'] = (tmp['age'] > 52).apply(lambda x: 1 if x else 0)
    tmp['gain_label1'] = (tmp['capital_gain'] == 0).apply(lambda x: 1 if x else 0)
    tmp['gain_label2'] = ((tmp['capital_gain'] < 4000) & (tmp['capital_gain'] > 0)).apply(lambda x: 1 if x else 0)
    tmp['gain_label3'] = ((tmp['capital_gain'] >= 4000) & (tmp['capital_gain'] < 7000)).apply(lambda x: 1 if x else 0)
    tmp['gain_label4'] = (tmp['capital_gain'] >= 7000).apply(lambda x: 1 if x else 0)

    columns = tmp.columns
    normal, scaler = standardize(tmp.values)
    tmp = pd.DataFrame(data=normal, columns=columns)
    
    tmp['age_gaussian'] = gaussian(tmp['age'])
    
    return tmp, scaler


def valid_preprocessing(data, scaler):
    tmp = data.copy(deep=True)
    tmp['gain2'] = tmp['capital_gain'] ** 2
    tmp['diff'] = tmp['capital_gain'] - tmp['capital_loss']
    tmp['new_age'] = np.absolute(1 / (tmp['age'] - 50.01))
    tmp['hpw'] = (tmp['hours_per_week'] > 39).apply(lambda x: 1 if x else 0)
    tmp['edu_num'] = (raw_trainX.iloc[valid_id].education_num.values) ** 2

#     tmp['age_label'] = (tmp['age'] > 52).apply(lambda x: 1 if x else 0)
    tmp['gain_label1'] = (tmp['capital_gain'] == 0).apply(lambda x: 1 if x else 0)
    tmp['gain_label2'] = ((tmp['capital_gain'] < 4000) & (tmp['capital_gain'] > 0)).apply(lambda x: 1 if x else 0)
    tmp['gain_label3'] = ((tmp['capital_gain'] >= 4000) & (tmp['capital_gain'] < 7000)).apply(lambda x: 1 if x else 0)
    tmp['gain_label4'] = (tmp['capital_gain'] >= 7000).apply(lambda x: 1 if x else 0)
        
    columns = tmp.columns
    tmp = standardize(tmp.values, scaler)
    tmp = pd.DataFrame(data=tmp, columns=columns)
    
    tmp['age_gaussian'] = gaussian(tmp['age'])
    
    return tmp


np.random.seed(5)
data_num = len(raw_X_train)
permute = np.random.permutation(data_num)
valid_id = permute[ : int(data_num * 0.1)]
train_id = permute[int(data_num * 0.1) :]

all_trainX = raw_X_train
# all_trainX = raw_trainX
all_trainY = raw_Y_train
# all_trainY = raw_trainY

trainX = all_trainX.iloc[train_id].copy(deep=True).reset_index(drop=True)
trainY = all_trainY.iloc[train_id].copy(deep=True).reset_index(drop=True)
validX = all_trainX.iloc[valid_id].copy(deep=True).reset_index(drop=True)
validY = all_trainY.iloc[valid_id].copy(deep=True).reset_index(drop=True)

X, scaler = train_preprocessing(trainX)
Y = trainY
X_ = valid_preprocessing(validX, scaler)
Y_ = validY

selected_names = X.columns


def sigmoid(x):
    tmp = 1.0 / (1.0 + np.exp(-x))
    return np.clip(tmp, 1e-8, 1-(1e-8))


def gradient_descient_adagrad(X, Y, X_, Y_, selected_names, learning_rate=0.01, batch_size=100, epoch=50, sample=False):
    Xtrain = X[selected_names]
    Ytrain = Y
    Xvalid = X_[selected_names]
    Yvalid = Y_
    
    data_num = len(Xtrain)
    feature_num = len(Xtrain.columns)
    batch_num = int(data_num / batch_size) + 1
    
    parameters = np.zeros(feature_num)
    bias = 0
    gradient_root = np.ones((feature_num + 1), dtype='float64') * 1e-8
    
    best = 0
    data_all = Xtrain.values
    for e in range(epoch):
        for batch in range(batch_num):
            if sample:
                stochastic = np.random.randint(len(Xtrain), size=batch_size)
            else:
                start = batch * batch_size
                if batch != batch_num - 1:
                    stochastic = list(range(start, start + batch_size))
                else:
                    stochastic = list(range(start, data_num))

            # CALCULATE LOSS AND GRADIENT
            data_matrix = data_all[stochastic]

            Ypredict = sigmoid(np.dot(data_matrix, parameters) + bias)
            Ytrue = Ytrain.iloc[stochastic].values

            loss = - (Ytrue * np.log(Ypredict) + (1 - Ytrue) * np.log(1 - Ypredict)).mean()
            Ydiff = - (Ytrue - Ypredict)

            gradient = np.dot(Ydiff, data_matrix) / len(Xtrain)
            gradient = np.append(gradient, Ydiff.mean())
            gradient_root = np.sqrt(gradient_root ** 2 + gradient ** 2)

            parameters -= learning_rate * gradient[:-1] / gradient_root[:-1]
            bias -= learning_rate * gradient[-1] / gradient_root[-1]
        
            probability = sigmoid(np.dot(Xvalid.values, parameters) + bias)
            Y_predict = (probability > 0.5).astype('int')
            valid = accuracy_score(Yvalid, Y_predict)
            
            if valid > best:
                best = valid
                print('loss: ' + str(loss) + ' | valid: ' + str(valid))
                best_parameters = copy.deepcopy(parameters)
                best_bias = copy.deepcopy(bias)
    
    return best_parameters, best_bias


parameters, bias = gradient_descient_adagrad(X, Y, X_, Y_, selected_names, learning_rate=0.01, batch_size=100, epoch=100, sample=False)

np.save('logistic_model/logistic_p.npy', parameters)
np.save('logistic_model/logistic_b.npy', bias)
np.save('logistic_model/logistic_m.npy', scaler[0])
np.save('logistic_model/logistic_s.npy', scaler[1])

print('Model seved. Use hw2_logistic.sh to make predictions.')
