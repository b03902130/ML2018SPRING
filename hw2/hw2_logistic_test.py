import sys
import copy
import numpy as np
import pandas as pd

test_path = sys.argv[1]
testX_path = sys.argv[2]
output_path = sys.argv[3]


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


def test_preprocesing(data, scaler):
    tmp = data.copy(deep=True)
    tmp['gain2'] = tmp['capital_gain'] ** 2
    tmp['diff'] = tmp['capital_gain'] - tmp['capital_loss']
    tmp['new_age'] = np.absolute(1 / (tmp['age'] - 50.01))
    tmp['hpw'] = (tmp['hours_per_week'] > 39).apply(lambda x: 1 if x else 0)
    tmp['edu_num'] = (raw_testX.education_num.values) ** 2
    
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


def sigmoid(x):
    tmp = 1.0 / (1.0 + np.exp(-x))
    return np.clip(tmp, 1e-8, 1-(1e-8))


raw_testX = pd.read_csv('test.csv')
raw_X_test = pd.read_csv('test_X.csv')

parameters = np.load('logistic_p.npy')
bias = np.load('logistic_b.npy')
mean = np.load('logistic_m.npy')
std = np.load('logistic_s.npy')
scaler = (mean, std)

X_test = test_preprocesing(raw_X_test, scaler)
selected_names = X_test.columns

X_selected = X_test[selected_names]
probability = sigmoid(np.dot(X_selected.values, parameters) + bias)
Y_predict = (probability > 0.5).astype('int')
pd.DataFrame(data={'id':np.arange(len(Y_predict))+1, 'label':Y_predict}).to_csv(output_path, index=False)
