import unittest
import numpy as np
from scipy.spatial.distance import cdist

class MyTestCase(unittest.TestCase):
    def test_check_clustering_method(self):
        method = 'kmeans'
        self.assertEqual(method, 'kmeans')

    def test_check_clustering_method_raise(self):
        method = 'Kmeans'
        with self.assertRaises(Exception):
            self.assertEqual(method, 'kmeans')

    def test_empirical_risk(self):
        X_block = np.array([[0, 0], [1, 0], [4, 4]])
        centroids_b = np.array([[0, 0], [4, 4]])
        emp_risk = cdist(X_block, centroids_b, 'sqeuclidean').min(axis=1).mean()
        self.assertTrue(isinstance(emp_risk, float))

    def test_median_impair(self):
        list_risks = [3, 2, 1, 4, 3, 5, 4]
        res = list_risks[np.argsort(list_risks)[len(list_risks) // 2]]
        self.assertEqual(res, np.median(list_risks))


if __name__ == '__main__':
    unittest.main()
