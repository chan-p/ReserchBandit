import numpy as np

class Random:
    def __init__(self, dimension):
        self.dimension = dimension

    def decide(self, userID, pool_articles):
        maxprob = float('-inf')
        maxid = None
        for id_, article in pool_articles.items():
            prob = self.get_prob(article)
            if maxprob < prob:
                maxprob = prob
                maxid = id_
        return maxid

    def get_prob(self, article):
        return np.random.random()

    def update(self, userID, article_feature, click):
        return

class LinUCBUserStruct:
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

class LinUCBAlgorithm:
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


if __name__=='__main__':
    random = Random(6)
    data = {'109511': [1.0, 0.381149, 0.000129, 0.060038, 0.269129, 0.289554], '109495': [1.0, 0.313277, 0.000125, 0.018413, 0.410555, 0.25763], '109503': [1.0, 0.306008, 0.00045, 0.077048, 0.230439, 0.386055], '109484': [1.0, 0.438513, 3e-06, 0.030714, 0.384494, 0.146277], '109502': [1.0, 0.277121, 0.000131, 0.038153, 0.335835, 0.34876], '109453': [1.0, 0.421669, 1.1e-05, 0.010902, 0.309585, 0.257833], '109473': [1.0, 0.295442, 1.4e-05, 0.135191, 0.292304, 0.27705], '109506': [1.0, 0.264355, 1.2e-05, 0.037393, 0.420649, 0.277591], '109509': [1.0, 0.306008, 0.00045, 0.077048, 0.230439, 0.386055], '109498': [1.0, 0.306008, 0.00045, 0.077048, 0.230439, 0.386055], '109513': [1.0, 0.211406, 3.6e-05, 0.002773, 0.569886, 0.2159], '109508': [1.0, 0.264355, 1.2e-05, 0.037393, 0.420649, 0.277591], '109515': [1.0, 0.281649, 0.000173, 0.195994, 0.151003, 0.371182], '109501': [1.0, 0.249086, 0.001009, 0.514682, 0.067732, 0.167491], '109514': [1.0, 0.29775, 1.3e-05, 0.011603, 0.512182, 0.178452], '109512': [1.0, 0.297322, 2.5e-05, 0.034951, 0.413566, 0.254137], '109505': [1.0, 0.375829, 2.5e-05, 0.033041, 0.349637, 0.241468], '109492': [1.0, 0.33183, 2.2e-05, 0.019904, 0.44039, 0.207855], '109494': [1.0, 0.306008, 0.00045, 0.077048, 0.230439, 0.386055], '109510': [1.0, 0.287909, 2.5e-05, 0.008983, 0.511333, 0.191751]}
    print(random.decide(data))
