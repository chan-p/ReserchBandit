from sklearn.cluster import KMeans
from sklearn.externals import joblib
from input_data import InputData

class UserCluster:
    def __init__(self, num_cluster):
        self.num_cluster = num_cluster
        self.model = KMeans(n_clusters=num_cluster, max_iter=500, tol=1e-04, random_state=0)

    def fit(self, features):
        self.model.fit(features)

    def predict_cluster(self, feature):
        return self.model.predict(feature)

    def model_load(self, model_file):
        return joblib.load('/Users/chan-p/GitHub/ReserchBandit/system/model/' + model_file)


if __name__=='__main__':
    km = UserCluster(25)
    users = []
    with open('../data/itemdata.csv') as f:
        for line in f:
            line = line.split(',')
            users.append(list(map(float, line[1:])))
    km.fit(users)
    joblib.dump(km, './model/item50_25.pkl')
