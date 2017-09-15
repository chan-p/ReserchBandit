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
    n = 1000
    km = UserCluster(n)
    users = []
    with open('../../../Desktop/R6/ydata-fp-td-clicks-v1_0.20090501') as f:
        for line in f:
            timestamp, click_article_id, click, user_data, article_pool = InputData.split_data(line)
            users.append(user_data)
            if len(users) == 3000000:
                km.fit(users)
                joblib.dump(km, './model/model'+str(n)+'_'+str(len(users))+'.pkl')
                # k = km.model_load('model40.pkl')
                '''
                for user in users:
                    print(k.predict_cluster(user))
                '''
        am = UserCluster(n)
        am.fit(users)
        joblib.dump(am, './model/model'+str(n)+'_'+str(len(users))+'.pkl')
"""

if __name__=='__main__':
    km = UserCluster(50)
    users = []
    with open('../analytics/stdev/usercluster200.csv') as f:
        for line in f:
            users.append(list(map(float, line.split(','))))
    km.fit(users)
    joblib.dump(km, './model/stdevmodel200_50.pkl')
"""
