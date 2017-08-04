from input_data import InputData
import sys
from method import Random
from user_clustering import UserCluster

N_CLUSTERING = 20
FILE_DIR = '/Users/chan-p/Desktop/R6/'
# FILE_DIR = '/Users/t-hayshi/Desktop/R6/'

reward = 0
count = 0
file_path = ['ydata-fp-td-clicks-v1_0.20090501']


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
    return kmeans

def evaluate(article, decide_article, click):
    global reward
    global count

    if decide_article == article:
        reward += click
        count += 1
        if count % 100 == 0:print(reward/count)
        return True
    return False

def run_enviroment(algorithms, cluster_model):
    try:
        ite = 0
        for file_name in file_path:
            with open(FILE_DIR + file_name) as f:
                for line in f:
                    timestamp, click_article_id, click, user_data, article_pool = InputData.split_data(line)
                    for name, alg in algorithms.items():
                        decide_id = alg.decide(article_pool)
                        if evaluate(click_article_id, decide_id, click): alg.update()
                    ite += 1
                    if ite % 10000 == 0: print(ite, count)
    except:
        ex, ms, tb = sys.exc_info()
        print("ErrorMessage:" + ms)
        print(ite)
        print(line)
    return


if __name__ == "__main__":

    dimension = 6

    # 手法の呼び出し
    algorithms = {}
    algorithms['Random'] = Random(dimension)

    cluster_model = user_clustering()
    print("=====Enviroment Start=====")
    # run_enviroment(algorithms, cluster_model)
