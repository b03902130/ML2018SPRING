import sys
import pandas as pd
import numpy as np
import copy

test_path = sys.argv[1]
testX_path = sys.argv[2]
output_path = sys.argv[3]


raw_testX = pd.read_csv(test_path)
raw_X_test = pd.read_csv(testX_path)

mean = np.load('generative_model/mean.npy')
std = np.load('generative_model/std.npy')
scaler = (mean, std)

mu1 = np.load('generative_model/mu1.npy')
mu2 = np.load('generative_model/mu2.npy')
shared_sigma = np.load('generative_model/shared_sigma.npy')
N1 = np.load('generative_model/N1.npy')
N2 = np.load('generative_model/N2.npy')


def standardize(data, scaler=0):
    if scaler == 0:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            scaler = (mean, std)
            return (data - mean) / std, scaler
    else:
            mean, std = scaler
            return (data - mean) / std


def valid_test_preprocessing(data, scaler):
    tmp = data.copy(deep=True)
    return standardize(tmp, scaler)


def sigmoid(x):
    tmp = 1.0 / (1.0 + np.exp(-x))
    return np.clip(tmp, 1e-8, 1-(1e-8))


def predict(X_valid, mu1, mu2, shared_sigma, N1, N2):
    inverse = np.linalg.inv(shared_sigma)
    w = np.dot(mu1 - mu2, inverse)
    b = -0.5 * np.dot(np.dot(mu1, inverse), mu1) + 0.5 * np.dot(np.dot(mu2, inverse), mu2) + np.log(N1 / N2)
    probability = sigmoid(np.dot(X_valid, w) + b)
    Y_predict = (probability > 0.5).astype(np.int)
    return Y_predict


X_test = valid_test_preprocessing(raw_X_test, scaler)
selected_names = X_test.columns

X_selected = X_test[selected_names]
Y_predict = predict(X_selected, mu1, mu2, shared_sigma, N1, N2)
pd.DataFrame(data={'id':np.arange(len(Y_predict))+1, 'label':Y_predict}).to_csv(output_path, index=False)
