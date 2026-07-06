from collections.abc import Callable

import numpy as np
from scipy.spatial import distance


def dist_matrix(x):
    return distance.cdist(x, x)

def elbow(x, x_cluster_labels, n_clusters, dist: Callable = dist_matrix):
    '''
    Parameters
    ----------
    x - data
    x_cluster_labels - data cluster labels
    n_clusters - cluster count
    dist - pairwise distance function
    
    Returns
    -------
    sum of squares of within cluster distances
    '''
    dists_sum = 0
    for i_cluster in range(n_clusters):
        x_i_cluster = x[x_cluster_labels == i_cluster]
        dists = dist(x_i_cluster)
        upper_sum = np.triu(dists ** 2).sum()
        dists_sum += upper_sum
    return dists_sum


def gap_statistic(
        x, x_cluster_labels, 
        clusterizer, n_clusters, nrefs=3, 
        dist: Callable = dist_matrix):
    '''
    Parameters
    ----------
    x - data
    x_cluster_labels - data cluster labels
    n_clusters - cluster count
    nrefs - number of generated uniform distribution reference samples.
            The mean within-cluster sum of squares is computed across all nrefs samples.
    dist - pairwise distance function
    '''
    dists_sum_uniform = []
    for _ in range(nrefs):
        # points from a uniform distribution
        ref_uniform = np.random.uniform(low=x.min(axis=0), high=x.max(axis=0), size=x.shape)
        # clustering points of uniform
        preds = clusterizer(ref_uniform)
        # sum of squares of within cluster distances for uniform clustering
        dists_sum_uniform.append(elbow(ref_uniform, preds, n_clusters, dist=dist))

    # sum of squares of within cluster distances for our clustering
    dists_sum = elbow(x, x_cluster_labels, n_clusters, dist=dist)
    gap = np.log(np.mean(dists_sum_uniform)) - np.log(dists_sum)
    return gap