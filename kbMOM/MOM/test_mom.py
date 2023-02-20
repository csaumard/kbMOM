import unittest
import numpy as np
from itertools import islice


class MyTestCase(unittest.TestCase):

    def test_doubleloop(self):
        init_blocks = [[0, 2, 2], [1, 2, 0]]
        blocks = [[1, 1, 1], [0, 0, 0]]

        mytest = [[b[j] for j in blocks[_]] for _, b in enumerate(init_blocks)]

        self.assertEqual(mytest, [[2, 2, 2], [1, 1, 1]])

    def test_doubleloop_2(self):
        data = np.array([10, 3, 5, 6])
        init_blocks = [[0, 2, 2], [1, 2, 0]]
        blocks = [[1, 1, 1], [0, 0, 0]]

        mytest = [[data[b[j]] for j in blocks[_]] for _, b in enumerate(init_blocks)]

        self.assertEqual(mytest, [[5, 5, 5], [3, 3, 3]])

    def test_slicing(self):
        input_list = [1, 2, 3, 4, 5, 6, 7]
        slicein = 3
        if len(input_list) % slicein == 0:
            length_to_split = [len(input_list) // slicein] * slicein
        else:
            length_to_split = [len(input_list) // slicein] * (slicein - 1)
            length_to_split.append(len(input_list) - (slicein - 1) * len(input_list) // slicein)
        lst = iter(input_list)
        res = [list(islice(lst, elem)) for elem in length_to_split]
        self.assertEqual(res, [[1, 2], [3, 4], [5, 6, 7]])

    def test_slicing_mean(self):
        data = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
        init_blocks = [[6, 3, 1], [0, 2, 5], [7, 9, 4, 8]]
        res = [np.mean(data[b]) for b in init_blocks]
        self.assertEqual(res, [0.0, 0.0, 1.0])

    def test_median_block_impair(self):
        list_means = [-1, 3, -3, 1, 0]
        Bm1 = len(list_means) - 1
        res = list_means[np.argsort(list_means)[Bm1 // 2]]
        self.assertEqual(res, 0)

    def test_median_block_pair(self):
        list_means = [-2, 3, -3, 2, 1, 0]
        Bm1 = len(list_means) - 1
        res = list_means[np.argsort(list_means)[Bm1 // 2]]
        self.assertEqual(res, 0)


if __name__ == '__main__':
    unittest.main()
