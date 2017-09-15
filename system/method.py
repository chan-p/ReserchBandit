import numpy as np
import math
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

class Random:
    def __init__(self, dimension):
        self.dimension = dimension

    def decide(self, userID, user_data, pool_articles):
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

    def update(self, userID, user_data, article_feature, click):
        return


class LinUCBUserStruct:
    def __init__(self, feature_dim, lambda_):
        self.d = feature_dim
        self.A = lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.AInv = np.linalg.inv(self.A)
        self.user_theta = np.random.rand(self.d)
        self.sigma = None
        self.time = 1
        self.click_time = 0

    def update_parameters(self, article_feature, click):
        self.A += np.outer(article_feature, article_feature)
        self.b += article_feature * click
        self.AInv = np.linalg.inv(self.A)
        self.user_theta = np.dot(self.AInv, self.b)
        self.time += 1
        if click == 1: self.click_time += 1

    def get_prob(self, alpha, article_feature):
        if alpha == -1:
            alpha = 0.1 * np.sqrt(np.log(self.time+1))
        mean = np.dot(self.user_theta, article_feature)
        var = np.sqrt(np.dot(np.dot(article_feature, self.AInv), article_feature))
        pta = mean + alpha * var
        return pta

    def save_weight(self, filename):
        with open('./weight/' + filename) as f:
            for userID in range(len(self.users)):
                for idx, weight in enumerate(self.users[userID].w):
                    f.write(str(weight))
                    if idx != 5:
                        f.write(',')
                    else:
                        f.write('\n')
                for idx, weight in enumerate(self.users[userID].sigma.reshape(1, 36)):
                    f.write(weight)
                    if idx != 36:
                        f.write(',')
                    else:
                        f.write('\n')

class LinUCBAlgorithm:
    def __init__(self, dim, alpha, lambda_, n):
        self.users = []

        for i in range(n):
            self.users.append(LinUCBUserStruct(dim, lambda_))

        self.dim = dim
        self.alpha = alpha

    def decide(self, userID, user_data, pool_articles):
        maxprob = float('-inf')
        maxid = None
        for id_, article in pool_articles.items():
            article = np.array(article)
            prob = self.users[userID].get_prob(self.alpha, article)
            if maxprob < prob:
                maxprob = prob
                maxid = id_
        return maxid

    def decide_try(self, userID, user_data, pool_articles):
        maxprob = float('-inf')
        maxid = None
        for id_, article in pool_articles.items():
            article = np.array(article)
            prob = self.users[userID].get_prob(self.alpha, article)
            if maxprob < prob:
                maxprob = prob
                maxid = id_
        return maxid

    def update(self, userID, user_data, article_feature, click, _):
        article_feature = np.array(article_feature)
        self.users[userID].update_parameters(article_feature, click)

    def get_diag(self, userID):
        return np.diag(self.users[userID].AInv)

    def get_CTR(self):
        ctr = 0.
        for userID in range(len(self.users)):
            user = self.users[userID]
            ctr += user.click_time / user.time
        return ctr / len(self.users)


    def save_weight(self, filename):
        with open('./weight/' + filename, 'w') as f:
            for userID in range(len(self.users)):
                for idx, weight in enumerate(list(self.users[userID].b)):
                    f.write(str(weight))
                    if idx != 5:
                        f.write(',')
                    else:
                        f.write('\n')
                for idx, weight in enumerate(list(self.users[userID].A.reshape(1, 36)[0])):
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
                    self.users[userID].b = np.array(vec)
                elif count % 2 == 0:
                    matrix = []
                    for val in line:
                        matrix.append(float(val))
                    self.users[userID].A = np.array(matrix).reshape(6, 6)
                    userID += 1
                count += 1


class LinUCBAlgorithm_Uni:
    def __init__(self, dim, alpha, lambda_, n):
        self.users = LinUCBUserStruct(dim, lambda_)

        self.dim = dim
        self.alpha = alpha

    def decide(self, userID, user_data, pool_articles):
        maxprob = float('-inf')
        maxid = None
        for id_, article in pool_articles.items():
            article = np.array(article)
            prob = self.users.get_prob(self.alpha, article)
            if maxprob < prob:
                maxprob = prob
                maxid = id_
        return maxid

    def decide_try(self, userID, user_data, pool_articles):
        maxprob = float('-inf')
        maxid = None
        for id_, article in pool_articles.items():
            article = np.array(article)
            prob = self.users.get_prob(self.alpha, article)
            if maxprob < prob:
                maxprob = prob
                maxid = id_
        return maxid

    def update(self, userID, user_data, article_feature, click, _):
        article_feature = np.array(article_feature)
        self.users.update_parameters(article_feature, click)


class CLUBUserStruct(LinUCBUserStruct):
    def __init__(self,featureDimension,  lambda_, userID):
        LinUCBUserStruct.__init__(self, featureDimension, lambda_)
        self.reward = 0
        self.CA = self.A
        self.Cb = self.b
        self.CAInv = np.linalg.inv(self.CA)
        self.CTheta = np.dot(self.CAInv, self.Cb)
        self.I = lambda_*np.identity(n = featureDimension)
        self.counter = 0
        self.CBPrime = 0
        self.d = featureDimension

    def updateParameters(self, articlePicked_FeatureVector, click, alpha_2):
        self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
        self.b += np.array(articlePicked_FeatureVector)*click
        self.AInv = np.linalg.inv(self.A)
        self.user_theta = np.dot(self.AInv, self.b)
        self.counter+=1
        self.CBPrime = alpha_2*np.sqrt(float(1+math.log10(1+self.counter))/float(1+self.counter))

    def updateParametersofClusters(self,clusters,userID,Graph,users):
        self.CA = self.I
        self.Cb = np.zeros(self.d)
        #print type(clusters)

        for i in range(len(clusters)):
            if clusters[i] == clusters[userID]:
                self.CA += float(Graph[userID,i])*(users[i].A - self.I)
                self.Cb += float(Graph[userID,i])*users[i].b
                self.CAInv = np.linalg.inv(self.CA)
                self.CTheta = np.dot(self.CAInv,self.Cb)

    def getProb(self, alpha, article_FeatureVector,time):
        mean = np.dot(self.CTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.CAInv),  article_FeatureVector))
        pta = mean +  alpha * var*np.sqrt(math.log10(time+1))
        return pta

class CLUBAlgorithm():
    def __init__(self,dimension,alpha,lambda_,n,alpha_2, cluster_init="Complete"):
        self.time = 0
        #N_LinUCBAlgorithm.__init__(dimension = dimension, alpha=alpha,lambda_ = lambda_,n=n)
        self.users = []
        #algorithm have n users, each user has a user structure
        for i in range(n):
            self.users.append(CLUBUserStruct(dimension,lambda_, i))

        self.dimension = dimension
        self.alpha = alpha
        self.alpha_2 = alpha_2
        if (cluster_init=="Erdos-Renyi"):
            p = 3*math.log(n)/n
            self.Graph = np.random.choice([0, 1], size=(n,n), p=[1-p, p])
            self.clusters = []
            g = csr_matrix(self.Graph)
            N_components, components = connected_components(g)
        else:
            self.Graph = np.ones([n,n])
            self.clusters = []
            g = csr_matrix(self.Graph)
            N_components, components = connected_components(g)

        self.CanEstimateCoUserPreference = False
        self.CanEstimateUserPreference = False
        self.CanEstimateW = False
        self.CanEstimateV = False

    def decide_try(self,userID, user_data, pool_articles):
        self.users[userID].updateParametersofClusters(self.clusters,userID,self.Graph, self.users)
        maxPTA = float('-inf')
        articlePicked = None

        for id_, article in pool_articles.items():
            x_pta = self.users[userID].getProb(self.alpha, article[:self.dimension],self.time)
            # pick article with highest Prob
            if maxPTA < x_pta:
                articlePicked = id_
                featureVectorPicked = article[:self.dimension]
                picked = article
                maxPTA = x_pta
            self.time +=1
            return articlePicked

    def update(self, userID, user_data, article, click, _):
        self.users[userID].updateParameters(article[:self.dimension], click, self.alpha_2)

    def updateGraphClusters(self,userID, binaryRatio):
        n = len(self.users)
        for j in range(n):
            ratio = float(np.linalg.norm(self.users[userID].user_theta - self.users[j].user_theta,2))/float(self.users[userID].CBPrime + self.users[j].CBPrime)
            #print float(np.linalg.norm(self.users[userID].user_theta - self.users[j].user_theta,2)),'R', ratio
            if ratio > 1:
                ratio = 0
            elif binaryRatio == 'True':
                ratio = 1
            elif binaryRatio == 'False':
                ratio = 1.0/math.exp(ratio)
            #print 'ratio',ratio
            self.Graph[userID][j] = ratio
            self.Graph[j][userID] = self.Graph[userID][j]
        N_components, component_list = connected_components(csr_matrix(self.Graph))
        #print 'N_components:',N_components
        self.clusters = component_list
        return N_components

    def getLearntParameters(self, userID):
        return self.users[userID].user_theta


class CoLinUCBUserSharedStruct(object):
    def __init__(self, featureDimension, lambda_, userNum, W):
        self.currentW = np.identity(n = userNum)

        self.W = W
        self.userNum = userNum
        self.A = lambda_*np.identity(n = featureDimension*userNum)
        self.b = np.zeros(featureDimension*userNum)
        self.AInv =  np.linalg.inv(self.A)
        self.user_theta = np.zeros(shape = (featureDimension, userNum))
        self.CoTheta = np.zeros(shape = (featureDimension, userNum))

        self.BigW = np.kron(np.transpose(W), np.identity(n=featureDimension))
        self.CCA = np.dot(np.dot(self.BigW , self.AInv), np.transpose(self.BigW))

        self.alpha_t = 0.0
        self.sigma = 1.e-200   #Used in the high probability bound, i.e, with probability at least (1 - sigma) the confidence bound. So sigma should be very small
        self.lambda_ = lambda_

    def updateParameters(self, articlePicked, click,  userID, update):
        pass

    def getProb(self, alpha, article, userID):

        TempFeatureM = np.zeros(shape =(len(article), self.userNum))
        TempFeatureM.T[userID] = article
        M = TempFeatureM
        TempFeatureV = np.reshape(M.T, M.shape[0]*M.shape[1])

        mean = np.dot(self.CoTheta.T[userID], article)
        var = np.sqrt(np.dot(np.dot(TempFeatureV, self.CCA), TempFeatureV))

        self.alpha_t = 0.01*np.sqrt(np.log(np.linalg.det(self.A)/float(self.sigma * self.lambda_) )) + np.sqrt(self.lambda_)
        if math.isnan(self.alpha_t): self.alpha_t = 0.1
        #pta = mean + alpha * var    # use emprically tuned alpha
        pta = mean + self.alpha_t *var   # use the theoretically computed alpha_t

        return pta

class CoLinUCBAlgorithm:
    def __init__(self, dimension, alpha, lambda_, n, update='inv'):  # n is number of users
        self.updates = update #default is inverse. Could be 'rankone' instead
        W = self.constructAdjMatrix(0)[0]
        self.USERS = CoLinUCBUserSharedStruct(dimension, lambda_, n, W)
        self.dimension = dimension
        self.alpha = alpha
        self.W = W

        self.CanEstimateUserPreference = False
        self.CanEstimateCoUserPreference = False
        self.CanEstimateW = False
        self.CanEstimateV = False

    def decide_try(self, userID, user_data, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for id_, article in pool_articles.items():
            x_pta = self.USERS.getProb(self.alpha, article, userID)
            # pick article with highest Prob
            if maxPTA < x_pta:
                articlePicked = id_
                maxPTA = x_pta

        return articlePicked

    def update(self, userID, user_data, article, click, _, updates='Inv'):
        self.USERS.updateParameters(article, click, userID, updates)

    def constructGraph(self):
        # n = len(self.users)
        n = 200
        G = np.zeros(shape = (n, n))
        for ui in range(n):
            for uj in range(n):
                G[ui][uj] = np.random.random() * 2 - 1.
        return G

    def constructAdjMatrix(self, m):
        # n = len(self.users)
        n = 200
        G = self.constructGraph()
        W = np.zeros(shape = (n, n))
        W0 = np.zeros(shape = (n, n)) # corrupt version of W
        for ui in range(n):
            for uj in range(n):
                W[ui][uj] = G[ui][uj]
                sim = W[ui][uj] + np.random.normal(scale = 0.01) # corrupt W with noise
                if sim < 0:
                    sim = 0
                W0[ui][uj] = sim
            # find out the top M similar users in G
            if m>0 and m<n:
                similarity = sorted(G[ui], reverse=True)
                threshold = similarity[m]

                # trim the graph
                for i in range(n):
                    if G[ui][i] <= threshold:
                        W[ui][i] = 0;
                        W0[ui][i] = 0;
            W[ui] /= sum(W[ui])
            W0[ui] /= sum(W0[ui])
        return [W, W0]

    def getLearntParameters(self, userID):
        return self.USERS.user_theta.T[userID]

    def getCoTheta(self, userID):
        return self.USERS.CoTheta.T[userID]

    def getA(self):
        return self.USERS.A


class GOBLinSharedStruct:
    def __init__(self, featureDimension, lambda_, userNum, W):
        from scipy.linalg import sqrtm
        self.W = W
        self.userNum = userNum
        self.A = lambda_*np.identity(n = featureDimension*userNum)
        self.b = np.zeros(featureDimension*userNum)
        self.AInv = np.linalg.inv(self.A)

        self.theta = np.dot(self.AInv , self.b)
        self.STBigWInv = sqrtm( np.linalg.inv(np.kron(W, np.identity(n=featureDimension))))
        self.STBigW = sqrtm(np.kron(W, np.identity(n=featureDimension)))

    def updateParameters(self,  articlePicked, click,  userID, _):
        featureVectorM = np.zeros(shape =(len(articlePicked), self.userNum))
        featureVectorM.T[userID] = articlePicked
        M = featureVectorM
        featureVectorV = np.reshape(M.T, M.shape[0]*M.shape[1])

        CoFeaV = np.dot(self.STBigWInv, featureVectorV)
        CoFeaV = np.array(CoFeaV, dtype=np.float64)
        self.A += np.outer(CoFeaV, CoFeaV)
        self.b += click * CoFeaV

        self.AInv = np.linalg.inv(self.A)

        self.theta = np.dot(self.AInv, self.b)

    def getProb(self,alpha , article, userID):

        featureVectorM = np.zeros(shape =(len(article), self.userNum))
        featureVectorM.T[userID] = article
        M = featureVectorM
        featureVectorV = np.reshape(M.T, M.shape[0]*M.shape[1])

        CoFeaV = np.dot(self.STBigWInv, featureVectorV)

        mean = np.dot(np.transpose(self.theta), CoFeaV)
        a = np.dot(CoFeaV, self.AInv)
        var = np.sqrt( np.dot( np.dot(CoFeaV, self.AInv) , CoFeaV))

        pta = mean + alpha * var

        return pta

class GOBLinAlgorithm(CoLinUCBAlgorithm):
    def __init__(self, dimension, alpha, lambda_, n):
        W = self.constructAdjMatrix(0)[0]
        CoLinUCBAlgorithm.__init__(self, dimension, alpha, lambda_, n, W)
        self.USERS = GOBLinSharedStruct(dimension, lambda_, n, W)

    def getLearntParameters(self, userID):
        thetaMatrix =  matrixize(self.USERS.theta, self.dimension)
        return thetaMatrix.T[userID]

    def constructGraph(self):
        # n = len(self.users)
        n = 200
        G = np.zeros(shape = (n, n))
        for ui in range(n):
            for uj in range(n):
                G[ui][uj] = np.random.random() * 2 - 1.
        return G

    def constructAdjMatrix(self, m):
        # n = len(self.users)
        n = 200
        G = self.constructGraph()
        W = np.zeros(shape = (n, n))
        W0 = np.zeros(shape = (n, n)) # corrupt version of W
        for ui in range(n):
            for uj in range(n):
                W[ui][uj] = G[ui][uj]
                sim = W[ui][uj] + np.random.normal(scale = 0.01) # corrupt W with noise
                if sim < 0:
                    sim = 0
                W0[ui][uj] = sim
            # find out the top M similar users in G
            if m>0 and m<n:
                similarity = sorted(G[ui], reverse=True)
                threshold = similarity[m]

                # trim the graph
                for i in range(n):
                    if G[ui][i] <= threshold:
                        W[ui][i] = 0;
                        W0[ui][i] = 0;
            W[ui] /= sum(W[ui])
            W0[ui] /= sum(W0[ui])
        return [W, W0]


class Hybrid_LinUCB_singleUserStruct(LinUCBUserStruct):
    def __init__(self, userFeature, lambda_, userID):
        LinUCBUserStruct.__init__(self, len(userFeature), lambda_)
        self.d = len(userFeature)

        self.B = np.zeros([self.d, self.d**2])
        self.userFeature = userFeature

    def vectorize(self,M):
        return np.reshape(M.T, M.shape[0]*M.shape[1])

    def updateParameters(self, articlePicked_FeatureVector, click):
        additionalFeatureVector = self.vectorize(np.outer(self.userFeature, articlePicked_FeatureVector))
        LinUCBUserStruct.update_parameters(self, np.array(articlePicked_FeatureVector), click)
        self.B +=np.outer(articlePicked_FeatureVector, additionalFeatureVector)

    def updateTheta(self, beta):
        self.user_theta = np.dot(self.AInv, (self.b- np.dot(self.B, beta)))


class Hybrid_LinUCBUserStruct:
    def __init__(self, featureDimension,  lambda_, userFeatureList):

        self.k = featureDimension**2
        self.A_z = lambda_*np.identity(n = self.k)
        self.b_z = np.zeros(self.k)
        self.A_zInv = np.linalg.inv(self.A_z)
        self.beta = np.dot(self.A_zInv, self.b_z)
        self.users = []

        for i in range(len(userFeatureList)):
            self.users.append(Hybrid_LinUCB_singleUserStruct(userFeatureList[i], lambda_ , i))

    def vectorize(self,M):
        return np.reshape(M.T, M.shape[0]*M.shape[1])

    def updateParameters(self, articlePicked_FeatureVector, click, userID):
        z = self.vectorize( np.outer(self.users[userID].userFeature, articlePicked_FeatureVector))

        temp = np.dot(np.transpose(self.users[userID].B), self.users[userID].AInv)

        self.A_z += np.dot(temp, self.users[userID].B)
        self.b_z +=np.dot(temp, self.users[userID].b)

        self.users[userID].updateParameters(articlePicked_FeatureVector, click)

        temp = np.dot(np.transpose(self.users[userID].B), self.users[userID].AInv)

        self.A_z = self.A_z + np.outer(z,z) - np.dot(temp, self.users[userID].B)
        self.b_z =self.b_z+ click*z - np.dot(temp, self.users[userID].b)
        self.A_zInv = np.linalg.inv(self.A_z)

        self.beta =np.dot(self.A_zInv, self.b_z)
        self.users[userID].updateTheta(self.beta)

    def getProb(self, alpha, article_FeatureVector,userID):
        x = article_FeatureVector
        z = self.vectorize(np.outer(self.users[userID].userFeature, article_FeatureVector))
        temp =np.dot(np.dot(np.dot(self.A_zInv, np.transpose(self.users[userID].B)), self.users[userID].AInv), x)
        mean = np.dot(self.users[userID].user_theta,  x)+ np.dot(self.beta, z)
        s_t = np.dot(np.dot(z, self.A_zInv),  z) + np.dot(np.dot(x, self.users[userID].AInv),  x)
        -2* np.dot(z, temp)+ np.dot(np.dot( np.dot(x, self.users[userID].AInv) ,  self.users[userID].B ) ,temp)

        var = np.sqrt(s_t)
        pta = mean + alpha * var
        return pta


class Hybrid_LinUCBAlgorithm(object):
    def __init__(self, dimension, alpha, lambda_,):
        userFeatureList = self.generateUserFeature(self.constructAdjMatrix(0)[0])
        self.dimension = dimension
        self.alpha = alpha
        self.USER = Hybrid_LinUCBUserStruct(dimension, lambda_, userFeatureList)

    def decide_try(self,userID, user_data, pool_articles):
        maxPTA = float('-inf')
        articlePicked = None

        for id_, article in pool_articles.items():
            x_pta = self.USER.getProb(self.alpha, article[:self.dimension], userID)
            if maxPTA < x_pta:
                articlePicked = id_
                maxPTA = x_pta
        return articlePicked

    def update(self, userID, user_data, article, click, _):
        self.USER.updateParameters(article, click, userID)

    def generateUserFeature(self,W):
        svd = TruncatedSVD(n_components=6)
        result = svd.fit(W).transform(W)
        return result

    def constructGraph(self):
        # n = len(self.users)
        n = 200
        G = np.zeros(shape = (n, n))
        for ui in range(n):
            for uj in range(n):
                G[ui][uj] = np.random.random() * 2 - 1.
        return G

    def constructAdjMatrix(self, m):
        # n = len(self.users)
        n = 200
        G = self.constructGraph()
        W = np.zeros(shape = (n, n))
        W0 = np.zeros(shape = (n, n)) # corrupt version of W
        for ui in range(n):
            for uj in range(n):
                W[ui][uj] = G[ui][uj]
                sim = W[ui][uj] + np.random.normal(scale = 0.01) # corrupt W with noise
                if sim < 0:
                    sim = 0
                W0[ui][uj] = sim
            # find out the top M similar users in G
            if m>0 and m<n:
                similarity = sorted(G[ui], reverse=True)
                threshold = similarity[m]

                # trim the graph
                for i in range(n):
                    if G[ui][i] <= threshold:
                        W[ui][i] = 0;
                        W0[ui][i] = 0;
            W[ui] /= sum(W[ui])
            W0[ui] /= sum(W0[ui])
        return [W, W0]
