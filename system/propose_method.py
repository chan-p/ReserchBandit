import numpy as np
import random
import math
import statistics
from user_clustering import UserCluster
from scipy.stats import norm
import copy


class OriginalUCBItemStruct:
    def __init__(self, dimension):
        self.d = dimension
        self.high_n = 0
        self.high_average_context = np.array([0. for i in range(self.d)])
        self.stdev_context = np.array([0.00001 for i in range(self.d)])
        self.ave_distance_click = 0.
        self.time = 0

        self.low_n = 0
        self.low_average_context = np.array([0. for i in range(self.d)])
        self.ave_distance_minus_click = 0.

    def update_stdev(self, article_feature, user):
        cos = lambda v1, v2 : np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        self.high_n += 1
        n = self.high_n - 1
        old = self.ave_distance_click
        self.ave_distance_click = (n*self.ave_distance_click + cos(self.high_average_context, user.user_theta[:self.d])) / float(n+1)
        if math.isnan(self.ave_distance_click): self.ave_distance_click = old
        for i in range(self.d):
            old_ave = self.high_average_context[i]
            self.high_average_context[i] = (n * self.high_average_context[i] + article_feature[i]) / float(n + 1)
            self.stdev_context[i] = np.sqrt((float(n) * (self.stdev_context[i]**2+old_ave**2) + article_feature[i]**2) / float(n+1) - self.high_average_context[i]**2)
            if math.isnan(self.stdev_context[i]): self.stdev_context[i] = 0.001
            # self.stdev_context[i] = 0.

    def update_low(self, article_feature, user):
        cos = lambda v1, v2 : np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        self.low_n += 1
        n = self.low_n - 1
        self.ave_distance_minus_click = (n*self.ave_distance_minus_click + cos(self.low_average_context, user.user_theta[:self.d])) / float(n+1)
        if math.isnan(self.ave_distance_click): self.ave_distance_click = 0.
        self.low_average_context += (n * self.low_average_context + article_feature[:self.d]) / float(n + 1)

    def get_CBF(self, item_list, user_theta, sim_dic):
        cos = lambda v1, v2 : np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        count_high = 0
        calc_high = cos(user_theta, self.high_average_context) * self.ave_distance_click
        for item in item_list:
            tmp_high = cos(user_theta, item.high_average_context) * item.ave_distance_click
            if math.isnan(tmp_high): tmp_high = 0.
            calc_high += tmp_high
            if tmp_high == 0.: count_high += 1
        calc_high = 0. if count_high > 3 else calc_high / float(len(item_list)+1 - count_high) # 9/4 1.242 左上上  1.255
        # calc_high = 0. if count_high > 4 else calc_high / float(len(item_list)+1 - count_high) # 右上 ダメ
        # calc_high = 0. if count_high > 2 else calc_high / float(len(item_list)+1 - count_high) # 右上 1.22
        if math.isnan(calc_high):
            return 0.
        else:
            return calc_high

    def get_itemvec(self, article):
        self.time += 1
        return np.append(article, self.stdev_context)


class OriginalUCBItemClusterStruct:
    def __init__(self, dimension):
        self.d = dimension
        self.A2 = 0.1*np.identity(dimension)
        self.b2 = np.zeros(dimension)
        self.A2Inv = np.linalg.inv(self.A2)
        self.item_cluster_theta = [random.random() for i in range(self.d)]

    def update(self, article, click, user):
        self.A2 += np.outer(user.user_theta[self.d*2:], user.user_theta[self.d*2:])
        self.b2 += user.user_theta[self.d*2:]*(click - user.user_theta[:self.d*2].dot(article[:self.d*2]))
        self.A2Inv = np.linalg.inv(self.A2)
        self.item_cluster_theta = np.dot(self.A2Inv, self.b2)

    def get_itemcluster_vec(self):
        return self.item_cluster_theta


class OriginalUCBUserStruct:
    def __init__(self, feature_dim, lambda_, id_):
        self.d = feature_dim
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.user_theta = np.random.rand(self.d) +  np.random.rand(self.d)
        # self.user_theta = [1. for i in range(self.d)]
        self.time = 1
        self.click_time = 0
        self.ID = id_
        self.ave_distance_click = 0.
        self.ave_stdev_reward = 0.

    def update_parameters(self, article_feature, click, user_data):
        self.A += np.outer(article_feature, article_feature)
        self.b += (article_feature) * (click)
        self.AInv = np.linalg.inv(self.A)
        self.user_theta = np.dot(self.AInv, self.b)
        if click == 1: self.click_time += 1

    def get_prob(self, alpha, article_feature, item_cluster):
        if alpha == -1:
            alpha = 0.1 * np.sqrt(np.log(self.time+1))
        mean = np.dot(self.user_theta, article_feature)
        var = np.sqrt(np.dot(np.dot(article_feature, self.AInv), article_feature))
        var2 = np.sqrt(np.dot(np.dot(self.user_theta[12:], item_cluster.A2Inv),  self.user_theta[12:]))
        pta = mean + alpha * var + alpha * var2
        return pta


class OriginalUCBAlgorithm:
    def __init__(self, dim, alpha, lambda_, num_user_cluster, num_woi_cluster):
        self.users = []
        self.items = []
        self.item_cluster = []
        self.latent = dim * 3
        self.d = dim

        for i in range(num_user_cluster):
            self.users.append(OriginalUCBUserStruct(self.latent, lambda_, i))

        for i in range(1000):
            self.items.append(OriginalUCBItemStruct(dim))

        for i in range(25):
            self.item_cluster.append(OriginalUCBItemClusterStruct(dim))

        self.dim = dim

        self.alpha = alpha
        self.dic = {}
        self.item_dic = {}
        self.now_vec = None
        self.item_count = 0
        self.sim_dic = self.clac_itemsim()
        self.sim_tuple = self.choice_itemsim()
        self.first_flg = False

        self.item_model = UserCluster(25).model_load('item50_25.pkl')

    def decide(self, userID, user_data, pool_articles):
        maxprob = -10000
        maxid = None
        for id_, article in pool_articles.items():
            id_ = int(id_)
            article = np.array(article)
            article_vec = copy.deepcopy(article)
            itemID = self.get_itemID(id_)
            # 新出のアイテムは類似度1位の追加コンテキストをコピー
            if self.items[itemID].time == 0:
                self.copy_vecs(id_, self.items[itemID])
            itemClusterID = self.get_itemClusterID(article)
            article = np.append(self.items[itemID].get_itemvec(article), self.item_cluster[itemClusterID].item_cluster_theta)
            prob_lin = self.users[userID].get_prob(self.alpha, article)
            prob = prob_lin
            if maxprob < prob:
                maxprob = prob
                maxid = id_
        return maxid

    def decide_try(self, userID, user_data, pool_articles):
        maxprob = -10000
        maxid = None

        for id_, article in pool_articles.items():
            id_ = int(id_)
            self.now_vec = article
            article = np.array(article)
            article_vec = copy.deepcopy(article)
            itemID = self.regist_itemID(id_, article)
            if self.items[itemID].time == 0:
                self.copy_vecs(id_, self.items[itemID])
            itemClusterID = self.get_itemClusterID(article)
            article = np.append(self.items[itemID].get_itemvec(article), self.item_cluster[itemClusterID].item_cluster_theta)
            prob_lin = self.users[userID].get_prob(self.alpha, article, self.item_cluster[itemClusterID])
            simIDs = self.get_itemsimID(self.sim_tuple[id_])
            prob_sim_item = self.items[itemID].get_CBF([self.items[simIDs[0]], self.items[simIDs[1]], self.items[simIDs[2]], self.items[simIDs[3]], self.items[simIDs[4]]], self.users[userID].user_theta[:6], self.sim_dic)
            # prob =  prob_lin # 追加ベクトル:clustervec,stdevvec 8/31現状一番良い 最終 1.253 最高 1.253
            prob = prob_lin + 0.8 * prob_sim_item # 追加ベクトル:clustervec,stdevvec + CBF的アプローチ(類似上位5までを使用) + 新出アイテムへのベクトルコピー 9/1現状一番良い 最終 1.272 最高 1.272 2回目：1.261 # 右上下
            if maxprob < prob:
                maxprob = prob
                maxid = id_
        return maxid

    def copy_vecs(self, itemID, item):
        for i in range(5):
            if self.items[self.get_itemID(self.sim_tuple[itemID][i])].high_n > 10:
                top_sim_item = self.items[self.get_itemID(int(self.sim_tuple[itemID][i]))]
                item.high_n = top_sim_item.high_n
                item.high_average_context = copy.deepcopy(top_sim_item.high_average_context)
                item.stdev_context = copy.deepcopy(top_sim_item.stdev_context)
                item.ave_distance_click = top_sim_item.ave_distance_click

    def get_itemID(self, article):
        if article not in self.dic:
            self.dic[article] = self.item_count
            self.item_count += 1
        return self.dic[article]

    def regist_itemID(self, article, article_vec):
        if article not in self.dic:
            self.dic[article] = self.item_count
            self.item_count += 1
            if len(self.dic) > 50:
                self.item_dic[article] = article_vec
                self.sim_dic = self.sim_calc()
                self.sim_tuple = self.choice_itemsim()
        return self.dic[article]

    def get_itemClusterID(self, article):
        return self.item_model.predict_cluster(article)[0]

    def get_itemsimID(self, sim_tuples):
        get_sim = lambda x : self.get_itemID(x)
        return list(map(get_sim, sim_tuples))

    def choice_itemsim(self):
        import copy
        sim_tuple = {}
        for id_, dics in self.sim_dic.items():
            keys = []
            values = []
            for id2_, sim in dics.items():
                keys.append(id2_)
                values.append(sim)
            values2 = copy.deepcopy(values)
            values.sort(reverse=True)
            sim_tuple[id_] = (keys[values.index(values2[0])], keys[values.index(values2[1])], keys[values.index(values2[2])], keys[values.index(values2[3])], keys[values.index(values2[4])])
        return sim_tuple

    def update(self, userID, user_data, article_feature, click, article_id):
        user_data = np.array(user_data)
        article_feature = np.array(article_feature)
        itemClusterID = self.get_itemClusterID(article_feature)
        article_feature = self.items[self.get_itemID(article_id)].get_itemvec(article_feature)
        article_feature = np.append(article_feature, self.item_cluster[itemClusterID].get_itemcluster_vec())

        self.users[userID].update_parameters(article_feature, click, user_data)
        self.item_cluster[itemClusterID].update(article_feature, click, self.users[userID])
        if int(click) == 1:
            # self.users[userID].update_stdev(article_feature)
            self.items[self.get_itemID(article_id)].update_stdev(article_feature, self.users[userID])

    def clac_itemsim(self):
        with open('../data/itemdata.csv') as f:
            for line in f:
                line = line.split(',')
                self.item_dic[line[0]] = []
                for val in line[1:]:
                    self.item_dic[line[0]].append(float(val))

                if int(line[0]) not in self.dic:
                    self.dic[int(line[0])] = self.item_count
                    self.item_count += 1
        return self.sim_calc()

    def sim_calc(self):
        sim_dic = {}
        cos = lambda v1, v2 : np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        for id_, val in self.item_dic.items():
            sim_dic[int(id_)] = {}
            id_ = int(id_)
            for id2_, val2 in self.item_dic.items():
                id2_ = int(id2_)
                if id_ == id2_: continue
                sim_dic[id_][id2_] = cos(val, val2)
                if math.isnan(sim_dic[id_][id2_]): sim_dic[id_][id2_] = 0.2
        return sim_dic

    def save_weight(self, filename):
        with open('./weight/' + filename, 'w') as f:
            for userID in range(len(self.users)):
                for idx, weight in enumerate(list(self.users[userID].w)):
                    f.write(str(float(weight)))
                    if idx != 5:
                        f.write(',')
                    else:
                        f.write('\n')
                for idx, weight in enumerate(list(self.users[userID].sigma.reshape(1, 36)[0])):
                    f.write(str(float(weight)))
                    if idx != 35:
                        f.write(',')
                    else:
                        f.write('\n')

    def load_weight(self, filename):
        count = 1
        userID = 0
        with open('./weight/' + filename) as f:
            for line in f:
                line = line.split(',')
                if (count-1) % 2 == 0:
                    vec = []
                    for val in line:
                        vec.append(float(val))
                    self.users[userID].w = np.array(vec)
                elif count % 2 == 0:
                    matrix = []
                    for val in line:
                        matrix.append(float(val))
                    self.users[userID].sigma = np.array(matrix).reshape(6, 6)
                    userID += 1
                count += 1

    def memory_item_num(self):
        with open('../data/num_item_click.csv', 'w') as f:
            for id_, item in enumerate(self.items):
                f.write(str(id_)+','+str(item.high_n)+'\n')
