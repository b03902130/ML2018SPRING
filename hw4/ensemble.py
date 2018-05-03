import numpy as np
import pandas as pd

import keras
from keras.optimizers import SGD
from keras.models import load_model

train_raw = pd.read_csv('train.csv').values
test_raw = pd.read_csv('test.csv').values

Xtrain_raw = np.array([np.fromstring(line, dtype=int, sep=' ') for line in train_raw[:, 1]])
Ytrain_raw = train_raw[:, 0].astype('int32')
Xtest_raw = np.array([np.fromstring(line, dtype=int, sep=' ') for line in test_raw[:, 1]])

Xtrain = Xtrain_raw.reshape((-1, 48, 48, 1))
Xtest = Xtest_raw.reshape((-1, 48, 48, 1))
Ytrain = Ytrain_raw
Ytrain = keras.utils.to_categorical(Ytrain, num_classes=7)

split = -1500
X = Xtrain[:split]
X_ = Xtrain[split:]
Y = Ytrain[:split]
Y_ = Ytrain[split:]


x1 = X_.astype(np.float32) / 255
x2 = (X_.astype(np.float32) - 128) / 255

model1 = load_model('model1.hdf5')
model2 = load_model('model2.hdf5')

input1 = keras.layers.Input(shape=(48,48,1))
X1 = model1(input1)
input2 = keras.layers.Input(shape=(48,48,1))
X2 = model2(input2)

out = keras.layers.Average()([X1, X2])
model = keras.models.Model(inputs=[input1, input2], outputs=out)

sgd = SGD(lr=0.005, decay=0.00001, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


valid = model.evaluate([x1, x2], Y_)
print('VALID ACC:', end=' ')
print(valid[1])


input1 = Xtest.astype(np.float32) / 255
input2 = (Xtest.astype(np.float32) - 128) / 255

predictions = model.predict([input1, input2]).argmax(axis=1)
pd.DataFrame(data={'id':range(len(predictions)), 'label':predictions}).to_csv('answer.csv', index=None)
