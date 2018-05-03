import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

images = np.load(sys.argv[1])
images = images.reshape(-1, 28, 28)

pca = PCA(n_components=300, whiten=True, svd_solver='full')
reducted = pca.fit_transform(images.reshape(-1, 28 * 28))

kmeans = KMeans(n_clusters=2, random_state=0)
clusterd = kmeans.fit_transform(reducted)
labeled = (clusterd[:, 0] > clusterd[:, 1]).astype(np.int32)

test_case = pd.read_csv(sys.argv[2]).values[:, 1:]
predictions = []
for case in test_case:
    if labeled[case[0]] == labeled[case[1]]:
        predictions.append(1)
    else:
        predictions.append(0)
predictions = np.array(predictions)

pd.DataFrame(data={'ID':range(len(predictions)), 'Ans':predictions}).to_csv(sys.argv[3], index=None, columns=['ID', 'Ans'])
