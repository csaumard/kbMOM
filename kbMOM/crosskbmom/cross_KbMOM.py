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

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from random import choices
import numpy as np


def check_clustering_method(method):
    """
    Check if the clustering method passed is supported by the procedure.
    Else propose the method to choose
    ```prms```
    # method      : str
    """
    list_methods = ['kmeans']
    if method in list_methods:
        return method
    else:
        raise ValueError("The method should be in " + list_methods)


class CrossKbMOM:

    def __init__(self, K, nbr_blocks, coef_ech, method='kmeans'):
        """
        Initialization of the class
        ```prms ```
        # K          : number of clusters
        # nbr_blocks : number of blocks for the subsampling
        # coef_ech   : number of datapoints in each block
        # method     : name of the method in {'kmeans', 'gmm'} by default, 'kmeans'
        """
        self.K = K
        self.B = nbr_blocks
        self.coef_ech = coef_ech
        self.method = check_clustering_method(method)

        # init output params
        self.centers = np.zeros((self.K,))
        self.risk = float
        self.list_block_risks = []

    def sampling_all_blocks_function(self, n):  # ,nbr_blocks,weighted_point,cluster_sizes):
        """
        # Creates nbr_blocks blocks based on self.coef_ech and self.B
        ```prms ```
        # n         : int, number of datapoints in my dataset

        ```return``` B of lists of coef_ech indices
        """
        blocks = [choices(np.arange(n), k=self.coef_ech) for i in range(self.B)]
        #blocks = [random.sample(np.arange(n), self.coef_ech) for i in range(self.B)]  # without replacement
        return blocks

    def one_block_clustering(self, data, block):
        """
        Cluster the sample created by the block dataset and retrieve the centroids
        ```prms```
        # block     : list, list of indices of data to consider

        ```return``` numpy array of size (K, p) with p the dimension of the data space
        """
        X_block = data[block, :]
        if self.method == 'kmeans':
            cls = KMeans(n_clusters=self.K, init='k-means++')
            cls.fit(X_block)
            return cls.cluster_centers_

    def one_block_risk(self, data, block, centroids_b):
        """
        Compute the empirical risk on the block b from centroids_b
        ```prms```
        # data       : dataset
        # block      : list, list of indices for the data to consider
        # centroid_b : numpy array, matrix of centroids

        ```Return``` float, Average of nearest distances to their closest centroids
        """
        X_block = data[block, :]
        return cdist(X_block, centroids_b, 'sqeuclidean').min(axis=1).mean()

    def one_block_robust_risk(self, data, index_b, centroids_b, blocks):
        """
        Compute the robust empirical risk of the block b with centroids_b
        ```prms```
        # data       : dataset
        # index_b    : int, index of the block to consider
        # centroid_b : numpy array, matrix of centroids

        ```Return``` float, Median of the empirical risks among all the blocks except index_b block
        """
        return self.get_median_risk([self.one_block_risk(data, blocks[b], centroids_b) for b in range(self.B)
                                     if b != index_b])

    def get_median_risk(self, list_risks):
        """
        Get the median risk among a list of risks (float)
        """
        Bm1 = self.B - 1
        return list_risks[np.argsort(list_risks)[Bm1 // 2]]

    def fit(self, data):
        """
        Main function which fits the data according to a robust estimation of the empirical risk
        ```parms```
        # data  : 2-d numpy array of size nxp, dataset

        ```Return```

        """
        n, p = data.shape

        # blocks creation
        blocks = self.sampling_all_blocks_function(n)

        # Loop Clustering one block
        list_centroids = [self.one_block_clustering(data, block) for block in blocks]

        # Loop Median empirical risks
        list_block_risks = [self.one_block_robust_risk(data, index_b, centroids_b, blocks) for index_b, centroids_b in
                            enumerate(list_centroids)]

        # Get the minimum among the median risks
        b_min = np.argmin(list_block_risks)

        # Output
        self.centers = list_centroids[b_min]
        self.risk = list_block_risks[b_min]
        self.list_block_risks = list_block_risks
        return self

    def predict(self, data):
        """
        Computes the partition based on the centroids of the cross Median Block
        ```parms```
        # data      : 2-d numpy array, dataset from which the partition is computed based on the robust centroids
        """
        D_nk = cdist(data, self.centers, 'sqeuclidean')
        return D_nk.argmin(axis=1)
