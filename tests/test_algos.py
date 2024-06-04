import numpy as np
import unittest

from robust_deconfounding.robust_regression import BFS, Torrent
from robust_deconfounding.decor import DecoR
from robust_deconfounding.utils import cosine_basis


class TestAlgos(unittest.TestCase):

    def helper(self, algo, n=100):
        np.random.seed(0)
        beta = np.array([[3.], [2.]])

        x = np.random.normal(0, 1, size=(n, 2))
        y = x @ beta + np.random.normal(0, 0.1, (n, 1))
        outliers = np.random.choice(n, int(0.1*n), replace=False)
        y[outliers] = np.random.normal(0, 1, (len(outliers), 1))

        algo.fit(x, y)

        self.assertEqual(algo.coef_.shape, (2,))
        self.assertLess(np.linalg.norm(algo.intercept_ - 0), 0.1)
        self.assertLess(np.linalg.norm(algo.coef_ - beta.T)/2, 0.1)

    def test_fit_tor(self):
        algo = Torrent(0.8, fit_intercept=True)
        self.helper(algo, 100)

    def test_fit_bfs(self):
        algo = BFS(0.8, fit_intercept=True)
        self.helper(algo, 10)

    def test_DecoR(self):
        np.random.seed(0)
        n = 100
        basis = cosine_basis(n)

        beta = np.array([[3.], [2.]])
        x = np.random.normal(0, 1, size=(n, 2))
        y = x @ beta + np.random.normal(0, 0.1, (n, 1))

        yn = basis.T @ y / n
        outliers = np.random.choice(n, int(0.1 * n), replace=False)
        yn[outliers] = np.random.normal(0, 1, (len(outliers), 1))

        y = basis @ yn

        algo = Torrent(0.8, fit_intercept=True)
        decor = DecoR(algo=algo, basis=basis)
        decor.fit(x, y)

        self.assertEqual(decor.estimate.shape, (2,))
        self.assertLess(np.linalg.norm(decor.estimate - beta.T)/2, 0.1)

