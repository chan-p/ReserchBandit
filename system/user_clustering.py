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
        return joblib.load('./model/' + model_file)


if __name__=='__main__':
    km = UserCluster(20)
    users = []
    with open('../../../Desktop/R6/ydata-fp-td-clicks-v1_0.20090501') as f:
        for line in f:
            timestamp, click_article_id, click, user_data, article_pool = InputData.split_data(line)
            users.append(user_data)
            if len(users) == 100:
                # km.fit(users)
                # joblib.dump(km, './model/model20.pkl')
                k = km.model_load('model20.pkl')
                for user in users:
                    print(k.predict_cluster(user))

