import numpy as np

class OriginalUCBUserStruct:
    def __init__(self, feature_dim, lambda_):
        self.d = feature_dim
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.user_theta = np.random.rand(self.d)
        self.time = 0

    def update_parameters(self, article_feature, click):
        self.A += np.outer(article_feature, article_feature)
        self.b += article_feature * click
        self.AInv = np.linalg.inv(self.A)
        self.user_theta = np.dot(self.AInv, self.b)
        self.time += 1

    def get_prob(self, alpha, article_feature):
        if alpha == -1:
            alpha = 0.1 * np.sqrt(np.log(self.time+1))
        mean = np.dot(self.user_theta, article_feature)
        var = np.sqrt(np.dot(np.dot(article_feature, self.AInv), article_feature))
        pta = mean + alpha * var
        return pta

class OriginalUCBAlgorithm:
    def __init__(self, dim, alpha, lambda_, n):
        self.users = []

        for i in range(n):
            self.users.append(LinUCBUserStruct(dim, lambda_))

        self.dim = dim
        self.alpha = alpha

    def decide(self, userID, pool_articles):
        maxprob = float('-inf')
        maxid = None
        for id_, article in pool_articles.items():
            article = np.array(article)
            prob = self.users[userID].get_prob(self.alpha, article)
            if maxprob < prob:
                maxprob = prob
                maxid = id_
        return maxid

    def update(self, userID, article_feature, click):
        article_feature = np.array(article_feature)
        self.users[userID].update_parameters(article_feature, click)

