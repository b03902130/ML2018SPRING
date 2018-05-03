import os
import sys
import numpy as np
from skimage import io
from numpy.linalg import svd

path = sys.argv[1]
images = []
for filename in os.listdir(path):
    filepath = path + '/' + filename
    images.append(io.imread(filepath))
images = np.stack(images)

def float2image(data):
    tmp = data.copy()
    tmp -= np.min(tmp)
    tmp /= np.max(tmp)
    tmp *= 255
    return tmp.reshape(600, 600, 3).astype(np.uint8)

means = images.reshape(415, -1).mean(axis=0)
image_features = images.reshape(415, -1) - means

u, s, v = svd(image_features.transpose(), full_matrices=False)

target = io.imread(path + '/' + sys.argv[2])

component_num = 4
preprocessed = target.reshape(-1) - means
coordinate = np.dot(preprocessed, u[:, :component_num])
reconstruction = means + np.dot(u[:, :component_num], coordinate)
reconstruction = float2image(reconstruction)

io.imsave("reconstruction.jpg", reconstruction)