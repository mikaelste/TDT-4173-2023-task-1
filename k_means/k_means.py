import numpy as np

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    def __init__(self, n_clusters=2, max_iters=100, cluster_assignments=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.cluster_assignments = cluster_assignments

    def fit(self, X):
        X = X.values
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            distances = cross_euclidean_distance(X, self.centroids)
            self.cluster_assignments = np.argmin(distances, axis=1)

            new_centroids = np.array([X[self.cluster_assignments == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        X = X.values
        distances = cross_euclidean_distance(X, self.centroids)
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments

    def get_centroids(self):
        return self.centroids

    # --- Attempt on auto custering ---
    def find_optimal_clusters(self, X, max_clusters=10):
        distortions = []
        for num_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=num_clusters, max_iters=self.max_iters)
            kmeans.fit(X)
            distortion = euclidean_distortion(X, kmeans.cluster_assignments)
            distortions.append(distortion)

        optimal_clusters = np.argmin(np.diff(distortions)) + 1

        return optimal_clusters

    def fit_auto(self, X, max_clusters=10):  # attempt on making an automatic fit
        optimal_clusters = self.find_optimal_clusters(X, max_clusters)
        self.n_clusters = optimal_clusters
        self.fit(X)

    # --- Attempt on auto custering end ---


# --- Some utility functions --- #


def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points

    Note: by passing "y=0.0", it will compute the euclidean norm

    Args:
        x, y (array<...,n>): float tensors with pairs of
            n-dimensional points

    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def cross_euclidean_distance(x, y=None):
    """ """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the raw distortion measure
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()

    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance

    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)

    Args:
        X (array<m,n>): m x n float matrix with datapoints
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    # Intra distance
    a = D[np.arange(len(X)), z]
    # Smallest inter distance
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))
