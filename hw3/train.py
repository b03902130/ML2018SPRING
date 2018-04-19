import sys
import numpy as np
import pandas as pd
from PIL import Image

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adam
from keras.models import load_model
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint


def fat(img):
    pre = img.resize((58, 48))
    pre = pre.crop((5, 0, 53, 48))
    return [np.array(pre)]

def thin(img):
    pre = img.resize((48, 58))
    pre = pre.crop((0, 5, 48, 53))
    return [np.array(pre)]

def zoom_in(img):
    pre = img.resize((58,58))
    result = []
    result.append(np.array(pre.crop((8, 8, 56, 56))))
    result.append(np.array(pre.crop((1, 1, 49, 49))))
    result.append(np.array(pre.crop((8, 1, 56, 49))))
    result.append(np.array(pre.crop((1, 8, 49, 56))))
    return result

def rotate(img):
    pre = img
    result = []
    result.append(np.array(pre.rotate(10)))
    result.append(np.array(pre.rotate(-10)))
    return result

def preprocessing(data):
    result = []
    for num, tmp in enumerate(data):
        tmp = tmp.reshape(48,48)
        result.append(tmp)
        
        result.append(np.pad(tmp[0:42,0:42], ((3,3),(3,3)), 'constant'))
        result.append(np.pad(tmp[6:48,0:42], ((3,3),(3,3)), 'constant'))
        result.append(np.pad(tmp[6:48,6:48], ((3,3),(3,3)), 'constant'))
        result.append(np.pad(tmp[0:42,6:48], ((3,3),(3,3)), 'constant'))
        
        img = Image.fromarray(tmp, mode='L')
        result += fat(img)
        result += thin(img)
        result += zoom_in(img)
        result += rotate(img)
                        
        flip_index = np.random.randint(13, size=6)
        for flip in flip_index:
            result[flip] = np.flip(result[flip], axis=1)
        
    return np.stack(result)


print('loading training data')
train_raw = pd.read_csv(sys.argv[1]).values
Xtrain_raw = np.array([np.fromstring(line, dtype=int, sep=' ') for line in train_raw[:, 1]])
Ytrain_raw = train_raw[:, 0].astype('int32')
Xtrain = Xtrain_raw.reshape((-1, 48, 48, 1)).astype(np.uint8)
Ytrain = Ytrain_raw
Ytrain = keras.utils.to_categorical(Ytrain, num_classes=7).astype(np.uint8)

print('preprocessing data')
split = -2870
X = preprocessing(Xtrain[:split]).reshape(-1, 48, 48, 1)
X_ = Xtrain[split:]
Y = np.vstack([np.tile(y, (13, 1)) for y in Ytrain[:split]])
Y_ = Ytrain[split:]

np.random.seed(5)
np.random.shuffle(X)
np.random.seed(5)
np.random.shuffle(Y)


print('defining model')
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units = 256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 7, activation='softmax'))


print('compiling model')
# opt = Adadelta()
opt = SGD(lr=0.005, decay=0.00001, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


checkpoint = ModelCheckpoint('model_{epoch:03d}_{val_acc:.3f}.hdf5',
                            monitor='val_acc',
                            verbose=0,
                            save_best_only=True,
                            mode='max')


def data_generater(X, Y, batch_size, epoch_num):
    for epoch in range(epoch_num):
        for batch_start in range(0, len(X), batch_size):
            if batch_start + batch_size < len(X):
                x_batch = X[batch_start : batch_start + batch_size]
                y_batch = Y[batch_start : batch_start + batch_size]
            else:
                x_batch = X[batch_start : len(X)]
                y_batch = Y[batch_start : len(Y)]
            x_batch = (x_batch.astype(np.float32) - 128) / 255
            
            yield x_batch, y_batch


def valid_data(X_, Y_):
    X_normal = (X_.astype(np.float32) - 128) / 255
    return X_normal, Y_


print('start training')
batches = 50
epochs = 100
spe = int(len(X) / batches) + 1
model.fit_generator(data_generater(X, Y, batches, epochs), steps_per_epoch=spe, epochs=epochs, validation_data=valid_data(X_, Y_), callbacks=[checkpoint])
