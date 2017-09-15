from input_data import InputData
import sys
import numpy as np
from sklearn.externals import joblib
from method import Random, LinUCBAlgorithm
from propose_method import OriginalUCBAlgorithm
from user_clustering import UserCluster

N_CLUSTERING = 200
FILE_DIR = '/Users/chan-p/Desktop/R6/'
# FILE_DIR = '/home/t-hayashi/Desktop/R6/'

reward = {}
count = {}
file_path = ['ydata-fp-td-clicks-v1_0.20090502', 'ydata-fp-td-clicks-v1_0.20090503']


def user_clustering():
    users = []
    with open(FILE_DIR + file_path[0]) as f:
        for line in f:
            _, _, _, user_data, _ = InputData.split_data(line)
            if user_data not in users: users.append(user_data)
            if len(users) % 10000 == 0: print(len(users))
    print(len(users))

    kmeans = UserCluster(N_CLUSTERING)
    kmeans.fit(features)
    joblib.dump(kmeans, 'kmeans.pkl')
    return kmeans

def evaluate(article, decide_article, click, name, ite, line):
    global reward
    global count

    if int(decide_article) == int(article):
        reward[name] += click
        count[name] += 1
        return True
    return False

def run_enviroment(algorithms, cluster_model):
    ite = 0
    for file_name in file_path:
        with open(FILE_DIR + file_name) as f:
            old = {}
            old['LinearedUCB'] = 0
            old['OriginalUCB'] = 0
            old_imp = {}
            old_imp['LinearedUCB'] = 0
            old_imp['OriginalUCB'] = 0
            click_count = 0
            click_eq_count = 0
            ctr_list = {}
            ctr_list['LinearedUCB'] = []
            ctr_list['OriginalUCB'] = []
            for line in f:
                _, click_article_id, click, user_data, article_pool=InputData.split_data(line)
                click_count += click
                userID = cluster_model.predict_cluster(user_data)[0]
                for name, alg in algorithms.items():
                    decide_id = alg.decide(userID, user_data, article_pool)
                    if evaluate(click_article_id, decide_id, click, name, ite, line):
                        # if click == 1: print(click, algorithms['OriginalUCB'].get_prob_check(userID, article_pool[decide_id]))
                        ctr_list[name].append((reward[name]-old[name])/(count[name]-old_imp[name]))
                        alg.update(userID, user_data, article_pool[decide_id], click)
                    if count[name] % 2000 == 0:
                        # print(ite, name, reward[name], count[name], reward[name]/count[name], reward[name]-old[name], count[name]-old_imp[name], (reward[name]-old[name])/(count[name]-old_imp[name]), click_count)
                        print(name, np.mean(ctr_list[name]), reward[name]-old[name], count[name]-old_imp[name], reward[name], count[name])
                        old[name] = reward[name]
                        old_imp[name] = count[name]
                        click_count = 0
                        click_eq_count = 0
                        ctr_list[name] = []
                        count[name] += 1
                        # alg.save_weight(name + '_weight_' + str(N_CLUSTERING) + '_3.csv')
                ite += 1
    return


if __name__ == "__main__":

    dimension = 6
    alpha = 0.3
    lambda_ = 0.1

    global reward
    global count

    # 手法の呼び出し
    algorithms = {}
    # algorithms['Random'] = Random(dimension)
    algorithms['LinearedUCB'] = LinUCBAlgorithm(dimension, alpha, lambda_, N_CLUSTERING)
    algorithms['OriginalUCB'] = OriginalUCBAlgorithm(dimension, alpha, lambda_, N_CLUSTERING)
    for name in algorithms.keys():
        algorithms[name].load_weight(name + '_weight_' + str(N_CLUSTERING) + '_3.csv')
        reward[name] = 0
        count[name] = 1

    # clus    ter_model = user_clustering()
    print("=====Enviroment Start=====")
    run_enviroment(algorithms, cluster_model=UserCluster(N_CLUSTERING).model_load('model'+str(N_CLUSTERING)+'_3.pkl'))
