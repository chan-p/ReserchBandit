import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys

path = os.path.join(os.path.dirname(__file__), '../system/')
sys.path.append(path)

from input_data import InputData
from user_clustering import UserCluster

clicks = {}

model = UserCluster(40).model_load('model40.pkl')
with open('/Users/chan-p/Desktop/R6/ydata-fp-td-clicks-v1_0.20090501') as f:
    for line in f:
        _, click_article_id, click, user_data, article_pool=InputData.split_data(line)
        userID = model.predict_cluster(user_data)[0]
        if click == 1:
            if userID not in clicks: clicks[userID] = []
            clicks[userID].append(user_data)
            if len(clicks[userID]) == 5000:
                break
            if len(clicks[userID]) % 100 == 0:
                print(userID)
                print(len(clicks[userID]))

for id_, val in clicks.items():
    print(id_)
    print(len(val))
    pca = PCA(n_components = 2)
    pca.fit(np.array(val))
    transformed = pca.fit_transform(np.array(val))
    plt.scatter(transformed[:, 0], transformed[:, 1])
    plt.show()
