import sys
import numpy as np
import pandas as pd
from keras.models import load_model

print(sys.argv[1])
test_raw = pd.read_csv(sys.argv[1]).values
Xtest_raw = np.array([np.fromstring(line, dtype=int, sep=' ') for line in test_raw[:, 1]])
Xtest = Xtest_raw.reshape((-1, 48, 48, 1))

print('loading')
path = 'ensemble_0.704.hdf5'
model = load_model(path)

input1 = Xtest.astype(np.float32) / 255
input2 = (Xtest.astype(np.float32) - 128) / 255

print('predicting')
predictions = model.predict([input1, input2]).argmax(axis=1)
pd.DataFrame(data={'id':range(len(predictions)), 'label':predictions}).to_csv(sys.argv[2], index=None)
