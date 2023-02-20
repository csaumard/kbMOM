# Date : 2023-02-15 10:57

# !/usr/bin/python
# -*- coding: utf-8 -*-
# """
# cross K-bMOM algorithm
# create a strategy independent from the clustering approach which is robust towards outliers or cluster of outliers
# The main patterns are found and well estimated by their centroid
# Note that we speak about outliers (which supposes 'small size' character) and not noisy data
# """

__author__ = "Camille Saumard"
__copyright__ = "Copyright 2023"
__version__ = "0.0"
__maintainer__ = "Camille Saumard"
__email__ = "camille.brunet@gmail.com"
__status__ = "version beta"

from itertools import islice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from random import choices


class Mom:

    def __init__(self, nbr_blocks, repetitions):
        """
        Initialization of the class
        ```prms ```
        # nbr_blocks : number of blocks for the subsampling
        """
        self.B = nbr_blocks
        self.repetitions = repetitions
        self.median_block_j = []

    def slicing(self, n):
        """
        ```prms```
        # n     : size of the 1-d dataset
        """
        if n % self.B == 0:
            length_to_split = [n // self.B] * self.B
        else:
            length_to_split = [n // self.B] * (self.B - 1)
            length_to_split.append(n - (self.B - 1) * n // self.B)
        return length_to_split

    def sampling_all_sublocks(self, block_init):  # ,nbr_blocks,weighted_point,cluster_sizes):
        """
        # Creates nbr_blocks blocks based on self.coef_ech and self.B with replacement
        ```prms ```
        # block_init  : list, list of indices

        ```return``` list, list of randomly selected indeices with replacement from block_init
        """
        # [choices(np.arange(n), k=size) for size in self.slicing(n)]
        return [choices(size, k=len(size)) for size in block_init]

    def one_block_mean(self, data, block):
        """
        Compute the empirical risk on the block b from centroids_b
        ```prms```
        # data       : dataset
        # block      : list, list of indices for the data to consider
        # centroid_b : numpy array, matrix of centroids

        ```Return``` float, Average of nearest distances to their closest centroids
        """
        return np.mean(data[block])

    def get_median_block(self, list_means):
        """
        Get the median risk among a list of risks (float)
        """
        Bm1 = self.B - 1
        return list_means[np.argsort(list_means)[Bm1 // 2]]

    def median_of_means(self, X):
        n = len(X)
        block_size = int(np.ceil(n/self.B))
        block_means = [np.mean(X[i:i+block_size]) for i in range(0, n, block_size)]
        return np.median(block_means)

    def generate_mom(self, nu, data_size):
        """
        Compute medians of means on a student_t distribution
        """
        list_means = []
        for i in range(self.repetitions):
            # Generate data
            X = np.random.standard_t(nu, data_size)
            list_means.append(np.sqrt(data_size)*self.median_of_means(X))
        return list_means

    def generate(self, data):
        n = len(data)
        block_size = int(np.ceil(n / self.B))
        # block initialisation
        init_blocks = [[j for j in range(i, i+block_size)] for i in range(0, n, block_size)]
        list_means = [self.one_block_mean(data, b) for b in init_blocks]
        # bootstrap
        for j in range(self.repetitions):
            blocks = self.sampling_all_sublocks(init_blocks)
            list_means_bootstrap = [self.one_block_mean(data, block) - block_mean for block, block_mean in zip(blocks, list_means)]
            self.median_block_j.append(np.sqrt(n) * np.median(list_means_bootstrap)) #self.get_median_block(list_means_bootstrap)


class Simulate:

    def __init__(self):
        pass

    def main_moms(self, nu, B, iterations, dsize):

        # Compute mom and boostrap mom
        cls = Mom(B, iterations)
        list_mom = cls.generate_mom(nu, dsize)

        plt.subplot(121)
        plt.hist(list_mom, range=(-15, 15), bins=20, density=True)
        plt.grid()
        plt.title('Distribution des MOMs  ' + str(np.mean(list_mom)))

        # Display the histogram of the real law
        reflaw = np.random.normal(loc=0.0, scale=np.sqrt(math.pi / 2 * nu / (nu - 2)), size=10000)
        plt.subplot(122)
        plt.hist(reflaw, bins=20, range=(-15, 15), alpha=0.5, density=True)
        plt.title('Vraie loi')
        plt.show()

    def bootstrap_mom(self, nu, B, iterations, dsize):
        # data generation
        data = np.random.standard_t(nu, dsize)

        # Compute mom and boostrap mom
        cls = Mom(B, iterations)
        cls.generate(data)
        list_mom = cls.generate_mom(nu, dsize)

        reflaw = np.random.normal(loc=0.0, scale=np.sqrt(math.pi / 2 * nu / (nu - 2)), size=iterations)
        df = pd.DataFrame({'MOM': list_mom, 'boostrapMOM': cls.median_block_j, 'Truth': reflaw})
        df.plot(kind='density')
        plt.grid()
        plt.title('Density plot for MOM and boostrap MOM')
        plt.show()


if __name__ == '__main__':
    Simulate().main_moms(nu=4, B=11, iterations=1000, dsize=100)
    Simulate().bootstrap_mom(nu=4, B=10, iterations=1000, dsize=100)
