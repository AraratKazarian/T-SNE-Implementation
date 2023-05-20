import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

mnist = fetch_openml('mnist_784', version = 1, as_frame = False)
mnist.target = mnist.target.astype(np.uint8)

X = pd.DataFrame(mnist["data"]).sample(n = 1000)
y = pd.DataFrame(mnist["target"]).loc[X.index]

X = PCA(n_components = 30).fit_transform(X)


class tsne:

    def __init__(self):
        pass

    def sigma_search(self, diff_i, i):

        result = np.inf
        norm = np.linalg.norm(diff_i, axis=1)
        std_norm = np.std(norm)
        for sigma in np.linspace(0.01 * std_norm, 5 * std_norm, 200):
            p = np.exp(-norm ** 2 / (2 * sigma ** 2))
            p[i] = 0
            p_new = np.maximum(p / np.sum(p), np.nextafter(0, 1))
            h = -np.sum(p_new * np.log2(p_new))
            if np.abs(np.log(self.perp) - h * np.log(2)) < np.abs(result):
                result = np.log(self.perp) - h * np.log(2)
                sigma_ = sigma
        return sigma_

    def original_pairwise_affinities(self, X):

        n = len(X)
        p_ij = np.zeros(shape=(n, n))
        for i in range(0, n):
            diff = X[i] - X
            sigma_i = self.sigma_search(diff, i)
            norm = np.linalg.norm(diff, axis=1)
            p_ij[i, :] = np.exp(-norm ** 2 / (2 * sigma_i ** 2))
            np.fill_diagonal(p_ij, 0)
            p_ij[i, :] = p_ij[i, :] / np.sum(p_ij[i, :])
        p_ij = np.maximum(p_ij, np.nextafter(0, 1))

        return p_ij

    def symmetric_p_ij(self, p_ij):

        n = len(p_ij)
        p_ij_symmetric = np.zeros(shape=(n, n))
        for i in range(0, n):
            for j in range(0, n):
                p_ij_symmetric[i, j] = (p_ij[i, j] + p_ij[j, i]) / (2 * n)
        e_ = np.nextafter(0, 1)
        p_ij_symmetric = np.maximum(p_ij_symmetric, e_)

        return p_ij_symmetric

    def initialization(self, X):

        return np.random.normal(loc=0, scale=1e-4, size=(len(X), self.n_dimensions))

    def low_dimensional_affinities(self, y):

        n = len(y)
        q_ij = np.zeros(shape=(n, n))
        for i in range(0, n):
            diff = y[i] - y
            norm = np.linalg.norm(diff, axis=1)
            q_ij[i, :] = (1 + norm ** 2) ** (-1)
        np.fill_diagonal(q_ij, 0)
        q_ij = q_ij / q_ij.sum()
        q_ij = np.maximum(q_ij, np.nextafter(0, 1))

        return q_ij

    def get_gradient(self, p_ij, q_ij, Y):
        n = len(p_ij)
        gradient = np.zeros(shape=(n, Y.shape[1]))
        for i in range(0, n):
            diff = Y[i] - Y
            A = np.array([(p_ij[i, :] - q_ij[i, :])])
            B = np.array([(1 + np.linalg.norm(diff, axis=1)) ** (-1)])
            C = diff
            gradient[i] = 4 * np.sum((A * B).T * C, axis=0)
        return gradient

    def fit(self, X, perplexity, num_iter, lr, early_exaggeration=4, n_dimensions=2):

        self.perp = perplexity
        self.num_iter = num_iter
        self.lr = lr
        self.n_dimensions = n_dimensions

        n = len(X)
        p_ij = self.original_pairwise_affinities(X)
        p_ij_symmetric = self.symmetric_p_ij(p_ij)
        Y = np.zeros(shape=(num_iter, n, self.n_dimensions))
        Y_minus1 = np.zeros(shape=(n, self.n_dimensions))
        Y[0] = Y_minus1
        Y1 = self.initialization(X)
        Y[1] = np.array(Y1)

        for t in range(1, self.num_iter - 1):
            if t < 250:
                a = 0.5
                early_exaggeration = early_exaggeration
            else:
                a = 0.8
                early_exaggeration = 1
            q_ij = self.low_dimensional_affinities(Y[t])
            gradient = self.get_gradient(early_exaggeration * p_ij_symmetric, q_ij, Y[t])
            Y[t + 1] = Y[t] - lr * gradient + a * (Y[t] - Y[t - 1])
            if t % 50 == 0 or t == 1:
                cost = np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))
                print(f"Iteration {t}: Value of Cost Function is {cost}")

        print(f"Final Cost Function = {np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))}")
        solution = Y[-1]

        return solution, Y
#
# model = tsne()
# model.fit(X, 10, 100, 200, 2)