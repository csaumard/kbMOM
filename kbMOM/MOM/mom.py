# Date : 2023-02-15 10:57

# !/usr/bin/python
# -*- coding: utf-8 -*-
# """
# Check the theoretical properties of the MOM estimators
# """

__author__ = "Camille Saumard"
__copyright__ = "Copyright 2023"
__version__ = "0.0"
__maintainer__ = "Camille Saumard"
__email__ = "camille.brunet@gmail.com"
__status__ = "version beta"

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

    def sampling_all_sublocks(self, block_init):
        """
        # resample with replacement indices in blocks based on block_init  = block of data indices
        ```prms ```
        # block_init  : list, list of indices

        ```return``` list, list of randomly selected indices with replacement from block_init
        """
        return [choices(size, k=len(size)) for size in block_init]

    def one_block_mean(self, data, block):
        """
        Compute the empirical mean on the block b from centroids_b
        ```prms```
        # data       : dataset
        # block      : list, list of indices for the data to consider

        ```Return``` float, Average of the data block
        """
        return np.mean(data[block])

    def get_median_block(self, list_means):
        """
        Get the median risk among a list of risks (float)
        ```prms```
        # list_means : list of block means

        ```Return``` float, Median of list_mean values
        """
        Bm1 = self.B - 1
        return list_means[np.argsort(list_means)[Bm1 // 2]]

    def median_of_means(self, X):
        """
        Compute the median of block means
        ```prms```
        # X : 1-d data

        ```Return``` float, Median of block average
        """
        n = len(X)
        block_size = int(np.ceil(n / self.B))
        block_means = [np.mean(X[i:i + block_size]) for i in range(0, n, block_size)]
        return np.median(block_means)

    def generate_mom(self, nu, data_size):
        """
        Compute self.repetitions times medians of means on a student_t distribution in order to get its distribution
        ```prms```
        # nu : Degree of freedom of the Student
        # data_size : length of data

        ```Return``` tuple, list of moms, list of data means
        """
        list_means = []
        list_moms = []
        for i in range(self.repetitions):
            # Generate data
            X = np.random.standard_t(nu, data_size)
            # get median of means of each block
            list_moms.append(self.median_of_means(X))
            # get the means of each block
            list_means.append(np.mean(X))
        return list_moms, list_means

    def generate_bootstrapMOM(self, data):
        """
        Generate bootstrap MOM according to the data blocks.
        ```prms```
        # data : 1-d dataset

        ```Return``` self.median_block_j list, list of medians of means of each boostrap block
        """
        n = len(data)
        block_size = int(np.ceil(n / self.B))

        # block initialisation
        init_blocks = [[j for j in range(i, i + block_size)] for i in range(0, n, block_size)]
        list_means = [self.one_block_mean(data, b) for b in init_blocks]

        # bootstrap
        for j in range(self.repetitions):
            blocks = self.sampling_all_sublocks(init_blocks)
            list_means_bootstrap = [self.one_block_mean(data, block) - block_mean for block, block_mean
                                    in zip(blocks, list_means)]
            self.median_block_j.append(np.median(list_means_bootstrap))


class Simulate:
    # Simulations to check the theoretical results on the distribution of Moms and bootstrap Moms
    def __init__(self, nu, B, iterations, dsize):
        """
        ```prms```
        # nu : degree of freedom of a student law
        # B  : nb of blokcs
        # iterations : repetitions of the experiment
        # dsize : length of simulated data
        """
        self.nu = nu
        self.B = B
        self.iterations = iterations
        self.dsize = dsize

    def main_moms(self):
        """
        Generate MOMs and visualiza its distribution
        ```prms```
        # nu : degree of freedom of a student law
        # B  : nb of blokcs
        # iterations : repetitions of the experiment
        # dsize : length of simulated data

        ```Return``` self.median_block_j list, list of medians of means of each boostrap block
        """
        cls = Mom(self.B, self.iterations)
        list_mom = cls.generate_mom(self.nu, self.dsize)

        reflaw = np.random.normal(loc=0.0, scale=np.sqrt(math.pi / 2 * self.nu / (self.nu - 2)), size=self.iterations)
        df = pd.DataFrame({'MOM': [np.sqrt(self.dsize) * x for x in list_mom[0]],
                           'N-density': reflaw})  # 'mean': [np.sqrt(dsize)*x for x in list_mom[1]],
        df.plot(kind='density')
        plt.grid()
        plt.title('Distribution des MOMs  ' + str(np.mean(list_mom)))
        plt.show()

    def bootstrap_mom(self):
        """
        Generate MOMs and visualize its distribution

        ```Return``` hist
        """
        # data generation
        data = np.random.standard_t(self.nu, self.dsize)

        # Compute mom and boostrap mom
        cls = Mom(self.B, self.iterations)

        # bootstrap mom
        cls.generate_bootstrapMOM(data)

        # mom
        list_mom = cls.generate_mom(self.nu, self.dsize)

        reflaw = np.random.normal(loc=0.0, scale=np.sqrt(math.pi / 2 * self.nu / (self.nu - 2)), size=self.iterations)
        df = pd.DataFrame({'MOM': [np.sqrt(self.dsize) * x for x in list_mom[0]],
                           'bootstrapMOM': [np.sqrt(self.dsize) * x for x in cls.median_block_j],
                           'Truth': reflaw})
        df.plot(kind='density')
        plt.grid()
        plt.title('Density plot for MOM and boostrap MOM')
        plt.show()


if __name__ == '__main__':
    Simulate(nu=4, B=5, iterations=1000, dsize=50).main_moms()
    Simulate(nu=4, B=5, iterations=1000, dsize=50).bootstrap_mom()
