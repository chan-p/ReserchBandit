import numpy as np

class OriginalUCBUserStruct:
    def __init__(self, feature_dim, lambda_):
        self.d = feature_dim
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.user_theta = np.random.rand(self.d)
        self.time = 0

        self.vias = 0.
        self.low_n = 0
        self.low_average_context = np.array([0. for i in range(self.d)])
        self.high_n = 0
        self.high_average_context = np.array([0. for i in range(self.d)])
        self.stdev_context = np.array([0. for i in range(self.d)])

    def update_parameters(self, article_feature, click, user_data):
        self.A += np.outer(article_feature, article_feature)
        self.b += article_feature * (click - self.vias - user_data)
        self.AInv = np.linalg.inv(self.A)
        self.user_theta = np.dot(self.AInv, self.b)
        self.time += 1
        self.vias = 0.

    def update_stdev(self, article_feature):
        n = self.high_n
        for i in range(self.d):
            old_ave = self.high_average_context[i]
            self.high_average_context[i] = (1./float(n+1)) * (n*self.high_average_context[i] + article_feature[i])
            self.stdev_context[i] = (float(n) * (self.stdev_context[i]+old_ave**2) + article_feature[i]**2) / float(n + 1) - self.high_average_context[i]**2

    def update_lowinterest(self, article_feature):
        self.low_n += 1
        n = self.low_n - 1
        for i in range(self.d):
            old_ave = self.low_average_context[i]
            self.low_average_context[i] = (1./float(n+1)) * (float(n)*self.low_average_context[i] + article_feature[i])

    def get_prob(self, alpha, user_data, article_feature):
        if alpha == -1:
            alpha = 0.1 * np.sqrt(np.log(self.time+1))
        mean = np.dot((self.user_theta + user_data), article_feature)
        var = np.sqrt(np.dot(np.dot(article_feature, self.AInv), article_feature))
        vias = 0
        vias = self.vias_calculation(article_feature)
        pta = mean + alpha * var + vias
        return pta

    def vias_calculation(self, article_feature):
        sim = 0
        de_sim = 0
        popular = 0
        sim = np.linalg.norm(self.high_average_context - article_feature)
        de_sim = np.linalg.norm(self.low_average_context - article_feature)

        self.vias = de_sim/20 - sim/20
        return self.vias

class OriginalUCBAlgorithm:
    def __init__(self, dim, alpha, lambda_, n):
        self.users = []

        for i in range(n):
            self.users.append(OriginalUCBUserStruct(dim, lambda_))

        self.dim = dim
        self.alpha = alpha

    def decide(self, userID, user_data, pool_articles):
        maxprob = float('-inf')
        maxid = None
        for id_, article in pool_articles.items():
            article = np.array(article)
            user = np.array(user_data)
            prob = self.users[userID].get_prob(self.alpha, user, article)
            if maxprob < prob:
                maxprob = prob
                maxid = id_
        return maxid

    def update(self, userID, user_data, article_feature, click):
        user_data = np.array(user_data)
        article_feature = np.array(article_feature)
        self.users[userID].update_parameters(article_feature, click, user_data)
        if click == 0:
            self.users[userID].update_lowinterest(article_feature)
        elif click == 1:
            self.users[userID].update_stdev(article_feature)

