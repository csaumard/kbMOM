# Robust clustering algorithms
2 versions of robust (to outliers) clustering algorithm are implemented

### KbMOM
> A first version of KbMOM is implemented 
The median-of-means is an estimator of the mean of a random variable
that has emerged as an efficient and flexible tool to design robust
learning algorithms with optimal theoretical guarantees. However,
its use for the clustering task suggests dividing the dataset into
blocks, which may provoke the disappearance of some clusters in some
blocks and lead to bad performances. To overcome this difficulty,
a procedure termed ``bootstrap median-of-means'' is proposed, where the blocks are generated with a replacement in the dataset. Considering the estimation of the mean of a random variable, the bootstrap median-of-means has a better breakdown point
than the median-of-means if enough blocks are generated. A clustering algorithm called K-bMOM is designed,
by performing Lloyd-type iterations together with the use of the bootstrap
median-of-means. Good performances are obtained on simulated and real-world
datasets for color quantization and an emphasis is put on the benefits of our
robust intialization procedure. On the theoretical side, K-bMOM is proven to achieve a  non-trivial breakdown point for well-clusterizable situations. Finally, by considering an idealized
version of the estimator, robustness is also tackled by deriving rates
of convergence for the K-means distortion in the adversarial contamination
setting. It is the first result of this kind for
the K-means distortion.

For further details :
> BRUNET-SAUMARD, Camille, GENETAY, Edouard, et SAUMARD, Adrien. K-bMOM: A robust Lloyd-type clustering algorithm based on bootstrap median-of-means. Computational Statistics & Data Analysis, 2022, vol. 167, p. 107370.

### cross KbMOM
The main idea here is to subsample the dataset in blocks and run a clustering algorithm in each block. The fitted centroids kept are those which minimise the median risks computed on all the blocks except the one where the centroids have been fitted. Therefore, the selected risk should (according to the theory) have a risk closed to the risk of the unknown law of the data.
