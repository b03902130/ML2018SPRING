import sys
import pandas as pd
import numpy as np
import copy

train_path = sys.argv[1]
trainX_path = sys.argv[2]
trainY_path = sys.argv[3]

# RAW DATA
raw_train = pd.read_csv(train_path )
raw_trainX = raw_train.iloc[:, :-1]
raw_trainY = raw_train.iloc[:, -1]

# ONE-HOT ENCODING
raw_X_train = pd.read_csv(trainX_path)
raw_Y_train = pd.read_csv(trainY_path, header=None).iloc[:, 0]


def standardize(data, scaler=0):
    if scaler == 0:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            scaler = (mean, std)
            return (data - mean) / std, scaler
    else:
            mean, std = scaler
            return (data - mean) / std


def train_preprocessing(data):
    tmp = data.copy(deep=True)
    tmp, scaler = standardize(tmp)
    return tmp, scaler


def valid_test_preprocessing(data, scaler):
    tmp = data.copy(deep=True)
    return standardize(tmp, scaler)


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
X_ = valid_test_preprocessing(validX, scaler)
Y_ = validY

selected_names = X.columns


def sigmoid(x):
    tmp = 1.0 / (1.0 + np.exp(-x))
    return np.clip(tmp, 1e-8, 1-(1e-8))


def cal_sigma(row, mu, sigma):
    diff = row - mu
    sigma += np.dot(diff.reshape(-1, 1), diff.reshape(1, -1))


def training(X_train, Y_train):
    # Gaussian distribution parameters
    data_num = X_train.shape[0]
    feature_num = X_train.shape[1]
    
    mu1 = X_train[Y_train == 1].mean().values
    mu2 = X_train[Y_train == 0].mean().values
    
    sigma1 = np.zeros((feature_num, feature_num))
    X_train[Y_train == 1].apply(lambda s: cal_sigma(s.values, mu1, sigma1), axis=1)
    sigma1 /= data_num
    sigma1 += np.eye(feature_num) * 1e-8
    
    sigma2 = np.zeros((feature_num, feature_num))
    X_train[Y_train == 0].apply(lambda s: cal_sigma(s.values, mu2, sigma2), axis=1)
    sigma2 /= data_num
    sigma2 += np.eye(feature_num) * 1e-8
    
    N1 = (Y_train == 1).sum()
    N2 = (Y_train == 0).sum()
    
    shared_sigma = sigma1 * N1 / data_num + sigma2 * N2 / data_num
    
    return mu1, mu2, shared_sigma, N1, N2


mu1, mu2, shared_sigma, N1, N2 = training(X, Y)

np.save('generative_model/mean.npy', scaler[0])
np.save('generative_model/std.npy', scaler[1])
np.save('generative_model/mu1.npy', mu1)
np.save('generative_model/mu2.npy', mu2)
np.save('generative_model/shared_sigma.npy', shared_sigma)
np.save('generative_model/N1.npy', N1)
np.save('generative_model/N2.npy', N2)

print('Model saved. Use hw2_generative.sh to make predictions.')
