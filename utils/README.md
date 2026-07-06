## Metrics

#### Unsupervised metrics (labels unknown)
- [**Silhouette coefficient**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [**Calinski-Harabasz Index**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html)<br>
  Ratio of the sum of between-cluster dispersions to within-cluster dispersions across all clusters
  (within-cluster dispersion — sum of squared distances from points to their cluster centers;
  between-cluster dispersion — sum of squared distances from cluster centers to the overall centroid,
  weighted by the number of points in each cluster).
- [**Davies-Bouldin Index**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html)<br>
  Average "similarity" between clusters, where similarity is a measure that compares
  the distance between clusters to the size of the clusters themselves.

#### Metrics for selecting the number of clusters
- **Silhouette coefficient**
- **Elbow method**<br>
  A plot of the total within-cluster sum of squares vs. the number of clusters is built.
  The bend in the curve indicates that additional clusters beyond a certain point provide little value.
- **Gap statistic**<br>
  The gap statistic compares the total within-cluster distances to their expected values
  under a null (typically uniform) reference distribution.
  The reference distribution is clustered, and the log ratio of its total within-cluster distances
  to those of the actual clustering is computed.
  The optimal number of clusters is the value at which the gap statistic is maximized,
  meaning the clustering structure is far from a random uniform distribution of points.