# Date : 2019-07-18 13:17

#!/usr/bin/python
# -*- coding: utf-8 -*-
#"""
# Some utils function to evaluate clustering:
# RMSE< mapping and accuracy
#"""

__author__ = "Camille Saumard"
__copyright__ = "Copyright 2019"
__version__ = "4.0"
__maintainer__ = "Camille Saumard"
__email__  = "camille.brunet@gmail.com"
__status__ = "version beta"

import numpy as np
import ot

from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

from scipy.stats import mode
from math import pi, log

def distortion(dat,centroids_hat):
    '''
    Compute the empirical distortion on data dat
    dat          : data
    centroid_hat : fitted centroids
    '''
    dists = cdist(dat,centroids_hat).min(axis=1)
    return(sum(dists**2))
    

def mapping(target_lbl,clusters):
    '''
    Dictionary which maps the number cluster to the most probable label
    true_label: list of true partition
    cluster   : fitted clusters
    K         : number of clusters
    '''
    mymap = {}
    for i,clu in enumerate(set(clusters)):
        mask = (clusters == clu)
        mymap[clu] = mode(target_lbl[mask])[0][0]
    return(mymap)
        
    
def RMSE(centers,centroids_hat,mapp):
    '''
    centers: theoritical centers
    centers_hat: fitted centroids
    '''
    k,p = centers.shape
    incr = 0
    for cluster, label in mapp.items():
        incr += sum((centroids_hat[int(cluster)] - centers[label])**2)
    return((incr/len(mapp))**0.5)

def accuracy(true_label,cluster,mapp):
    labels = [mapp[clu] for clu in cluster]
    return(accuracy_score(true_label, labels))


def wdist(classmb,centers,lbl,centroids):
    '''
    Compute Wasserstein Distance between centroids and centers
    classmb   : class membership (True labels)
    centers   : true centers
    lbl       : fitted labels
    centroids : centers fitted by the algorithm
    '''
    label1 = classmb.tolist()
    label2 = lbl.tolist()
    a = np.array([label1.count(k) for k in range(max(label1)+1)])
    a = a/a.sum() 
    b = np.array([label2.count(k) for k in range(max(label2)+1)])
    b = b/b.sum()
    
    centers1 = np.array(centers)
    centers2 = np.array(centroids)
    M = cdist(centers1,centers2) # compute distances between clustering results and generated centers
    
    return(ot.emd2(a=a,b=b,M=M))


def loglikelihood(X,centers):
    '''
    COmpute the loglik of data based on Q-centers of the last median block
    centers = np.array
    X       = dataset
    '''
    K,p = centers.shape
    n,p = X.shape
    
    # Compute distance matrix between X and the centroids
    Xdist     = cdist(X,centers,'sqeuclidean')
    Xdist_k   = Xdist.min(axis=1)
    partition = Xdist.argmin(axis=1)
    size_k    = np.bincount(partition) 
    
    # Compute the variance term sig2*Ip
    sig2 = 1/(n-K)*sum(Xdist_k)
    
    # Compute loglik 
    cste  =  log(1/((2*pi)**0.5)*sig2**p/2)#-   n/2*log(2*pi) - p/2*log(sig2)
    term1 = - sum(Xdist_k) / (2*sig2)
    term2 = 0
    for k in range(K):
        term2  += size_k[k]*log(size_k[k] / n)
    loglik = cste + term1 + term2
    return(loglik)

def BIC(X,centers):
    '''
    Function which compute BIC criterion
    centers = np.array
    X       = dataset
    '''
    n,p = X.shape 
    K,p = centers.shape
    
    #complexity = (p+1)*K #
    loglik     = loglikelihood(X,centers)
    bic        = loglik - ((K-1) + p*K + 1)/2 * log(n)
    return(bic)