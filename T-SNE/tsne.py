import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Fetching the MNIST dataset and preprocessing
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)

# Sampling 1000 data points for faster computation
X = pd.DataFrame(mnist["data"]).sample(n=1000)
y = pd.DataFrame(mnist["target"]).loc[X.index]

# Dimensionality reduction using PCA
X = PCA(n_components=30).fit_transform(X)

# t-SNE implementation
class tsne:
    def __init__(self):
        pass

    def sigma_search(self, diff_i, i):
        """
        Searches for the optimal sigma to achieve the desired perplexity.
        """
        result = np.inf
        norm = np.linalg.norm(diff_i, axis=1)
        std_norm = np.std(norm)

        # Testing a range of sigma values to minimize the cost
        for sigma in np.linspace(0.01 * std_norm, 5 * std_norm, 200):
            p = np.exp(-norm ** 2 / (2 * sigma ** 2))
            p[i] = 0  # Setting self-affinity to 0
            p_new = np.maximum(p / np.sum(p), np.nextafter(0, 1))
            h = -np.sum(p_new * np.log2(p_new))  # Computing entropy
            if np.abs(np.log(self.perp) - h * np.log(2)) < np.abs(result):
                result = np.log(self.perp) - h * np.log(2)
                sigma_ = sigma
        return sigma_

    def original_pairwise_affinities(self, X):
        """
        Computes the original pairwise affinities in high-dimensional space.
        """
        n = len(X)
        p_ij = np.zeros((n, n))
        for i in range(n):
            diff = X[i] - X
            sigma_i = self.sigma_search(diff, i)
            norm = np.linalg.norm(diff, axis=1)
            p_ij[i, :] = np.exp(-norm ** 2 / (2 * sigma_i ** 2))
            np.fill_diagonal(p_ij, 0)
            p_ij[i, :] = p_ij[i, :] / np.sum(p_ij[i, :])
        p_ij = np.maximum(p_ij, np.nextafter(0, 1))
        return p_ij

    def symmetric_p_ij(self, p_ij):
        """
        Computes symmetric pairwise affinities.
        """
        n = len(p_ij)
        p_ij_symmetric = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                p_ij_symmetric[i, j] = (p_ij[i, j] + p_ij[j, i]) / (2 * n)
        p_ij_symmetric = np.maximum(p_ij_symmetric, np.nextafter(0, 1))
        return p_ij_symmetric

    def initialization(self, X):
        """
        Initializes the low-dimensional embeddings randomly.
        """
        return np.random.normal(loc=0, scale=1e-4, size=(len(X), self.n_dimensions))

    def low_dimensional_affinities(self, y):
        """
        Computes the low-dimensional pairwise affinities.
        """
        n = len(y)
        q_ij = np.zeros((n, n))
        for i in range(n):
            diff = y[i] - y
            norm = np.linalg.norm(diff, axis=1)
            q_ij[i, :] = (1 + norm ** 2) ** (-1)
        np.fill_diagonal(q_ij, 0)
        q_ij = q_ij / q_ij.sum() 
        q_ij = np.maximum(q_ij, np.nextafter(0, 1))
        return q_ij

    def get_gradient(self, p_ij, q_ij, Y):
        """
        Computes the gradient for updating low-dimensional embeddings.
        """
        n = len(p_ij)
        gradient = np.zeros((n, Y.shape[1]))
        for i in range(n):
            diff = Y[i] - Y
            A = np.array([(p_ij[i, :] - q_ij[i, :])])
            B = np.array([(1 + np.linalg.norm(diff, axis=1)) ** (-1)])
            C = diff
            gradient[i] = 4 * np.sum((A * B).T * C, axis=0)
        return gradient

    def fit(self, X, perplexity, num_iter, lr, early_exaggeration=4, n_dimensions=2):
        """
        Fits the t-SNE model to the input data.
        """
        self.perp = perplexity
        self.num_iter = num_iter
        self.lr = lr
        self.n_dimensions = n_dimensions

        n = len(X)
        p_ij = self.original_pairwise_affinities(X)
        p_ij_symmetric = self.symmetric_p_ij(p_ij)
        Y = np.zeros((num_iter, n, self.n_dimensions))
        Y[0] = np.zeros((n, self.n_dimensions))
        Y[1] = self.initialization(X)

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

# Testing and Visualization
model = tsne()
embedding, _ = model.fit(X, perplexity=30, num_iter=300, lr=200)

# Plotting the results
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=y.values.flatten(), cmap='tab10', s=15)
plt.colorbar()
plt.title('t-SNE Visualization of MNIST Dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()