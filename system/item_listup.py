from input_data import InputData
import sys
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from method import Random, LinUCBAlgorithm
from propose_method import OriginalUCBAlgorithm
from user_clustering import UserCluster
import datetime

N_CLUSTERING = 200
FILE_DIR = '/Users/chan-p/Desktop/R6/'
# FILE_DIR = '/home/t-hayashi/Desktop/R6/'

reward = {}
count = {}
file_path = ['ydata-fp-td-clicks-v1_0.20090501', 'ydata-fp-td-clicks-v1_0.20090502', 'ydata-fp-td-clicks-v1_0.20090503','ydata-fp-td-clicks-v1_0.20090504', 'ydata-fp-td-clicks-v1_0.20090505', 'ydata-fp-td-clicks-v1_0.20090506', 'ydata-fp-td-clicks-v1_0.20090507', 'ydata-fp-td-clicks-v1_0.20090508', 'ydata-fp-td-clicks-v1_0.20090509', 'ydata-fp-td-clicks-v1_0.20090510']
def article_pool_10(article_pool, click_article_id):
    new_article_pool = {}
    pool = list(article_pool.keys())
    target_pool = list(set(pool) - set([click_article_id]))
    ids = random.sample(target_pool, 9)
    ids.append(click_article_id)
    for id_ in ids:
        new_article_pool[id_] = article_pool[id_]
    return new_article_pool

def run_enviroment(algorithms, cluster_model):
    iteration = 0
    ite = {}
    click_count = {}
    old = {}
    old2 = {}
    old_imp = {}
    old_imp2 = {}
    ctr = {}
    ctr_list = {}
    rew = {}
    writer = {}
    now = datetime.datetime.today()

    for name in algorithms.keys():
        old[name] = 0
        old_imp[name] = 0
        old2[name] = 0
        old_imp2[name] = 0
        ctr_list[name] = [1.]
        ctr[name] = []
        rew[name] = [0]
        ite[name] = 0
        click_count[name] = 1
        writer[name] = open('../data/'+name+'_experiment_log_'+str(now)+'.csv', 'w')
    p = open('../data/new_itemdata.csv', 'w')
    dic = []
    for file_name in file_path:
        with open(FILE_DIR + file_name) as f:
            for line in f:
                _, click_article_id, click, user_data, article_pool=InputData.split_data(line)
                article_pool = article_pool_10(article_pool, click_article_id)
                userID = cluster_model.predict_cluster(user_data)[0]
                # user_item_log[userID][]
                for id_, vec in article_pool.items():
                    if id_ not in dic:
                        dic.append(id_)
                        sentence = ','.join(map(str, vec))
                        p.write(str(id_)+','+str(sentence)+','+str(iteration)+'\n')
                if iteration % 10000 == 0: print(iteration)
                """
                for name, alg in algorithms.items():
                    ite[name] += 1
                    click_count[name] += click
                    decide_id = alg.decide_try(userID, user_data, article_pool)
                    if iteration < 150000 and random.random() < (1.- (iteration*0.5/150000)):
                        decide_id = random.choice(list(article_pool.keys()))
                    random_ctr = click_count[name]/ite[name]
                    cumulated_ctr = reward[name]/count[name]
                    relative_ctr = cumulated_ctr / random_ctr

                    if iteration % 2000 == 0:
                        ctr[name].append(np.mean(ctr_list[name]))
                        writer[name].write(str(float('{:.5f}'.format(np.mean(ctr[name]))))+','+str(float('{:.5f}'.format(np.mean(ctr_list[name]))))+','+str(float('{:.5f}'.format(cumulated_ctr)))+'\n')
                        if iteration % 10000 == 0:
                            print(iteration, name, float('{:.5f}'.format(np.mean(ctr[name]))), float('{:.5f}'.format(np.mean(ctr_list[name]))), reward[name]-old2[name], count[name]-old_imp2[name], reward[name], count[name], float('{:.5f}'.format(cumulated_ctr)), float('{:.5f}'.format(relative_ctr)))
                            old2[name] = reward[name]
                            old_imp2[name] = count[name]
                        rew[name].append(reward[name]-old[name])
                        old[name] = copy.deepcopy(reward[name])
                        old_imp[name] = copy.deepcopy(count[name])
                        ctr_list[name] = [1.]
                        count[name] += 1
                        # alg.save_weight(name + '_weight_' + str(N_CLUSTERING) + '_3.csv')

                    if evaluate(click_article_id, decide_id, click, name, ite, line):
                        ctr_list[name].append(relative_ctr)
                        alg.update(userID, user_data, article_pool[decide_id], click, decide_id)
                """
                iteration += 1
    # gragh(ctr, rew)
    """
    for name, alg in algorithms.items():
        print(name, np.mean(ctr[name]))
        writer[name].close()
        alg.memory_item_num()
    """
    p.close()
    return


if __name__ == "__main__":
    import time
    start = time.time()
    dimension = 6
    alpha = 0.3
    lambda_ = 0.1
    n_stdev_cluster = 50
    # model_file = 'model'+str(N_CLUSTERING)+'_3000000.pkl'
    model_file = 'model'+str(N_CLUSTERING)+'_3.pkl'
    global reward
    global count

    # 手法の呼び出し
    algorithms = {}
    # algorithms['Random'] = Random(dimension)
    # algorithms['LinearedUCB'] = LinUCBAlgorithm(dimension, alpha, lambda_, N_CLUSTERING)
    # algorithms['OriginalUCB'] = OriginalUCBAlgorithm(dimension, alpha, lambda_, N_CLUSTERING, n_stdev_cluster)
    # for name in algorithms.keys():
    #     reward[name] = 1
    #     count[name] = 1

    # cluster_model = user_clustering()
    print("=====Enviroment Start=====")
    run_enviroment(algorithms, cluster_model=UserCluster(N_CLUSTERING).model_load(model_file))

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
