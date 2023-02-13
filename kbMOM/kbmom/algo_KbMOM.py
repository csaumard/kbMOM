# Date : 2020-02-06 16:08

# !/usr/bin/python
# -*- coding: utf-8 -*-
# """
# K-bMOM algorithm
# """

__author__ = "Camille Saumard"
__copyright__ = "Copyright 2023"
__version__ = "5.0"
__maintainer__ = "Camille Saumard"
__email__ = "camille.brunet@gmail.com"
__status__ = "version beta"

import numpy as np
import random
from math import log
from scipy.spatial.distance import cdist

from .kmedianpp import kmedianpp_init


class KbMOM:

    def __init__(self, K, nbr_blocks, coef_ech=6, max_iter=10, outliers=None, confidence=0.95,
                 quantile=0.5, initial_centers=None, init_type='km++', averaging_strategy='cumul'):
        """
        Initialization of the class
        ```parms ``
        # K             : number of clusters
        # nbr_blocks    : number of blocks to create in init and loop
        # coef_ech      : NUMBER of data in each block and cluster
        # quantile      : quantile to keep for the empirical risk; by default the median
        # max_iter      : number of iterations of the algorithm
        # max_iter_init : number of iterations to run for the kmeans in the initilization procedure
        # kmeanspp      : boolean. If true then init by kmeanspp else kmedianpp
        # outliers      : number of supposed outliers`

        ```usage ```
        cls = KbMOM(K = 3, nbr_blocks = 50, coef_ech= 9, outliers= 10)
        cls.fit(X) # with X the numpy array of data
        """

        # given element
        self.K = K
        self.max_iter = max_iter
        self.quantile = quantile
        self.coef_ech = coef_ech
        self.B = nbr_blocks
        self.alpha = 1 - confidence
        self.outliers = outliers
        self.init_type = init_type
        self.averaging_strategy = averaging_strategy

        # Deal with exceptions:
        if self.coef_ech <= self.K:
            self.coef_ech = 2 * self.K

        if isinstance(initial_centers, np.ndarray):
            self.centers = initial_centers
        else:
            self.centers = 0

        self.block_empirical_risk = []
        self.median_block_centers = []
        self.empirical_risk = []
        self.iter = 1
        self.warnings = 'None'

    def breakdownpoint_check(self):
        """
        Test some given values
        """
        # Test some given values
        if self.outliers is not None:
            t_sup = self.bloc_size(self.n, self.outliers)
            if self.coef_ech > t_sup:
                self.coef_ech = max((t_sup - 5), 1)
                self.coef_ech = int(round(self.coef_ech))
                print('warning:: the size of blocks has been computed according to the breakdown point theory')

            B_sup = self.bloc_nb(self.n, self.outliers, b_size=self.coef_ech, alpha=self.alpha)
            if self.B < B_sup:
                self.B = round(B_sup) + 10
                self.B = int(self.B)
                print('warning:: the number of blocks has been computed according to the breakdown point theory')

    def init_centers_function(self, X, idx_blocks):
        """
        # Initialisation function: create nbr_blocks blocks, initialize with a kmeans++,
        retrieve the index of the median block and its empirical risk value

         ``` prms ```
        . X          : numpy array of data
        . idx_blocks : list of indices contained in the B blocks
        """
        block_inertia = []
        init_centers = []
        if self.init_type == 'km++':
            # instanciation of kmeans++
            x_squared = X ** 2
            x_squared_norms = x_squared.sum(axis=1)

            for idx_ in idx_blocks:
                init_centers_ = kmedianpp_init(X[idx_, :], self.K, x_squared_norms[idx_], n_local_trials=None,
                                               square=True)
                init_centers.append(init_centers_)
                block_inertia.append(self.inertia_function(idx_, init_centers_))
        else:
            for idx_ in idx_blocks:
                init_centers_ = self.random_init(X[idx_, :])
                init_centers.append(init_centers_)
                block_inertia.append(self.inertia_function(idx_, init_centers_))

        median_risk = sorted(block_inertia)[round(self.quantile * len(block_inertia))]

        # Select the Q-quantile bloc
        id_median = block_inertia.index(median_risk)

        # init centers
        self.centers = init_centers[id_median]

        return (id_median, median_risk)

    def random_init(self, dataset):
        """
        Select randomly K datapoints from the dataset
        ``` prms ```
        . dataset     : numpy array of data
        """
        return dataset[np.random.choice(len(dataset), self.K), :]

    def sampling_all_blocks_function(self):  # ,nbr_blocks,weighted_point,cluster_sizes):
        """
        # Creates nbr_blocks blocks based on self.coef_ech and self.B
        """
        blocks = [random.choices(np.arange(self.n), k=self.coef_ech) for i in range(self.B)]
        return blocks

    def inertia_function(self, idx_block, centroids=None):
        """
        # Function which computes empirical risk per block

         ``` prms ```
        . idx_block  : list of indices contained in the B blocks
        . centroids  : if not None get the centers from kmeans++ initialisation
        """
        if not isinstance(centroids, np.ndarray):
            centroids = self.centers

        X_block = self.X[idx_block, :]
        nearest_centroid = cdist(X_block, centroids, 'sqeuclidean').argmin(axis=1)

        if len(set(nearest_centroid)) == self.K and sum(np.bincount(nearest_centroid) > 1) == self.K:
            within_group_inertia = 0
            for k, nc in enumerate(set(nearest_centroid)):
                centers_ = X_block[nearest_centroid == nc, :].mean(axis=0).reshape(1, -1)
                within_group_inertia += cdist(X_block[nearest_centroid == nc, :], centers_, 'sqeuclidean').sum()

            return within_group_inertia / len(idx_block)
        else:
            return -1

    def median_risk_function(self, blocks):
        """
        Computes the sum of all within variances and return the index of the median block
        and its empirical risk

        ```parameters ```
            . blocks     : list of indices forming the blocks
        """

        block_inertia = list(map(self.inertia_function, blocks))

        nb_nonvalide_blocks = sum(np.array(block_inertia) == -1)
        nb_valide_blocks = int(self.B - nb_nonvalide_blocks)

        if nb_nonvalide_blocks != self.B:

            median_risk = sorted(block_inertia)[nb_nonvalide_blocks:][round(self.quantile * nb_valide_blocks)]

            # Select the Q-quantile bloc
            id_median = block_inertia.index(median_risk)
            return (id_median, median_risk)

        else:
            return (None, None)

    def medianblock_centers_function(self, X, id_median, blocks):
        """
        Compute the barycenter of each cluster in the median block

         ``` prms ```
         . blocks     : list of indices forming the blocks
         . X          : matrix of datapoints
         . id_median  : index of the median block
        """
        X_block = X[blocks[id_median], :]
        distances = cdist(X_block, self.centers, 'sqeuclidean')
        nearest_centroid = distances.argmin(axis=1)

        centers_ = np.zeros((len(set(nearest_centroid)), self.p))
        for k, nc in enumerate(set(nearest_centroid)):
            centers_[k, :] = X_block[nearest_centroid == nc, :].mean(axis=0)
        self.centers = centers_
        return self

    def weigthingscheme(self, median_block):
        """
        Function which computes data depth

        ``` prms ```
        . median_block: list containing the indices of data in the median block
        """
        for idk in median_block:
            self.score[idk] += 1
        return self

    def initialisation(self, X):
        """
            Initialize the kbMOM clustering
        """
        if not isinstance(self.centers, np.ndarray):
            idx_block = self.sampling_all_blocks_function()
            id_median, median_risk_ = self.init_centers_function(X, idx_block)

            # update
            self.block_empirical_risk.append(median_risk_)
            self.medianblock_centers_function(X, id_median, idx_block)
            self.median_block_centers.append(self.centers)
            self.empirical_risk.append(sum(cdist(X, self.centers, 'sqeuclidean').min(axis=1)) / self.n)
            self.weigthingscheme(median_block=idx_block[id_median])
        else:
            # the initial datapoints are given
            self.block_empirical_risk.append(-1)
            self.median_block_centers.append(self.centers)
            self.empirical_risk.append(sum(cdist(X, self.centers, 'sqeuclidean').min(axis=1)) / self.n)

    def fit(self, X):
        """
        # Main loop of the K-bmom algorithm:

         ``` prms ```
        . X          : matrix of datapoints
        """
        # internal init
        self.X = X
        self.n, self.p = X.shape
        self.score = np.ones((self.n,))
        self.breakdownpoint_check()

        # initialisation step
        self.initialisation(X)

        if self.averaging_strategy == 'cumul':
            cumul_centers_ = self.centers

            # Main Loop - fitting process
        if self.max_iter == 0:
            condition = False
        else:
            condition = True

        while condition:

            # sampling
            idx_block = self.sampling_all_blocks_function()

            # Compute empirical risk for all blocks and select the empirical-block
            id_median, median_risk_ = self.median_risk_function(X, idx_block)

            # If blocks are undefined, then restarting strategy
            loop_within = 0
            while (id_median is None) and loop_within < 10:
                idx_block = self.sampling_all_blocks_function()
                id_median, median_risk_ = self.init_centers_function(X, idx_block)
                cumul_centers_ = np.zeros((self.K, self.p))
                self.warnings = 'restart'
                loop_within += 1

            if id_median is None:
                self.iter = self.max_iter
                self.warnings = 'algorithm did not converge'
                condition = False

            else:
                # update all parameters
                self.block_empirical_risk.append(median_risk_)
                self.medianblock_centers_function(X, id_median, idx_block)
                self.median_block_centers.append(self.centers)
                self.empirical_risk.append(sum(cdist(X, self.centers, 'sqeuclidean').min(axis=1)) / self.n)
                self.weigthingscheme(median_block=idx_block[id_median])

                if self.averaging_strategy == 'cumul' and self.iter > (self.max_iter - 10):
                    decay = self.max_iter - 10
                    # current_centers = self.pivot(self.centers,cumul_centers_)
                    cumul_centers_ = (self.centers / (self.iter - decay)) + (self.iter - decay - 1) / (
                            self.iter - decay) * cumul_centers_
                    self.centers = cumul_centers_

                self.iter += 1
                if self.iter >= self.max_iter:
                    condition = False

        return self

    def predict(self, X):
        """
        Function which computes the partition based on the centroids of Median Block
        """
        D_nk = cdist(X, self.centers, 'sqeuclidean')
        return D_nk.argmin(axis=1)

    def bloc_size(self, n_sample, n_outliers):
        """
        Function which fits the maximum size of blocks before a the breakpoint
        ```prms```
        n_sample: nb of data
        n_outlier: nb of outliers
        """
        return log(2.) / log(1 / (1 - (n_outliers / n_sample)))

    def bloc_nb(self, n_sample, n_outliers, b_size=None, alpha=0.05):
        """
        Function which fits the minimum nb of blocks for a given size t before a the breakpoint
        ```prms```
        n_sample: nb of data
        n_outlier: nb of outliers
        b_size = bloc_size
        alpha : threshold confiance
        """
        if n_outliers / n_sample >= 0.5:
            print('too much noise')
            return ()
        elif b_size is None:
            t = self.bloc_size(n_sample, n_outliers)
            return log(1 / alpha) / (2 * ((1 - n_outliers / n_sample) ** t - 1 / 2) ** 2)
        else:
            t = b_size
            return log(1 / alpha) / (2 * ((1 - n_outliers / n_sample) ** t - 1 / 2) ** 2)

    def stopping_crit(self, risk_median):
        risk_ = risk_median[::-1][:3]
        den = (risk_[2] - risk_[1]) - (risk_[1] - risk_[0])
        Ax = risk_[2] - (risk_[2] - risk_[1]) ** 2 / den
        return Ax

    def stopping_crit_GMM(self, risk_median):
        risk_ = risk_median[::-1][:3]
        Aq = (risk_[0] - risk_[1]) / (risk_[1] - risk_[2])

        Rinf = risk_[1] + 1 / (1 - Aq) * (risk_[0] - risk_[1])
        return Rinf

    def pivot(self, mu1, mu2):
        error = cdist(mu1, mu2).argmin(axis=1)
        pivot_mu = np.zeros((self.K, self.p))
        for i, j in enumerate(error):
            pivot_mu[i, :] = mu1[j, :]
        return pivot_mu
