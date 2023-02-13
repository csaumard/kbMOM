''' from sklearn repository : https://github.com/scikit-learn/scikit-learn '''
''' Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. '''

import warnings
import numpy as np
import scipy.sparse as sp

# Pairwise distances
def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,X_norm_squared=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.
    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::
        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
    
    Parameters
    ----------
    X : {array-like}, shape (n_samples_1, n_features)
    Y : {array-like}, shape (n_samples_2, n_features)
    Y_norm_squared : array-like, shape (n_samples_2, ), optional
        Pre-computed dot-products of vectors in Y (e.g.,
        ``(Y**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.
    squared : boolean, optional
        Return squared Euclidean distances.
    X_norm_squared : array-like, shape = [n_samples_1], optional
        Pre-computed dot-products of vectors in X (e.g.,
        ``(X**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.
    Notes
    -----
    To achieve better accuracy, `X_norm_squared` and `Y_norm_squared` may be
    unused if they are passed as ``float32``.
    Returns
    -------
    distances : array, shape (n_samples_1, n_samples_2)
    Examples
    --------
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = [[0, 1], [1, 1]]
    >>> # distance between rows of X
    >>> euclidean_distances(X, X)
    array([[0., 1.],
           [1., 0.]])
    >>> # get distance to origin
    >>> euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])
    See also
    --------
    paired_distances : distances betweens pairs of elements of X and Y.
    """
    XX = row_norms(X, squared=True)[:, np.newaxis]

    if X is Y and XX is not None:
        # shortcut in the common case euclidean_distances(X, X)
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)

        if YY.shape != (1, Y.shape[0]):
            raise ValueError(
                "Incompatible dimensions for Y and Y_norm_squared")
        if YY.dtype == np.float32:
            YY = None
    elif Y.dtype == np.float32:
        YY = None
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    distances = - 2 * np.dot(X,Y.T)#safe_sparse_dot(X, Y.T, dense_output=True)
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    if X is Y:
        np.fill_diagonal(distances, 0)
    if squared:
        return distances
    else:
        return distances**0.5


def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.
    Equivalent to np.sqrt((X * X).sum(axis=1))
    
    Parameters
    ----------
    X : array_like
        The input array
    squared : bool, optional (default = False)
        If True, return squared norms.
    Returns
    -------
    array_like
        The row-wise (squared) Euclidean norm of X.
    """
        
    norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms


def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    axis : int, optional
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(np.isclose(out.take(-1, axis=axis), expected, rtol=rtol,
                             atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out


def kmedianpp_init(X, n_clusters, x_squared_norms, n_local_trials=None,square=True):
    """Init n_clusters seeds according to k-means++
    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).
    n_clusters : integer
        The number of seeds to choose
    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.
    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    square : if False then compute D(x) else compute D(x)**2
    
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    This version has been updated such as kmedianpp can be used
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = np.random.randint(n_samples, size=1)#random_state.randint(n_samples)

    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,squared=square)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=square)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers
