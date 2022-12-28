import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float

import sys

# Clustering Methods


def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        # YOUR CODE HERE
        new_centers = np.zeros(shape=(k, 2))
        for i in range(N):
            min_distance = 2147483647
            min_index = -1
            for j in range(k):
                dist = np.sqrt(np.linalg.norm(features[i] - centers[j]))
                if dist < min_distance:
                    min_distance = dist
                    min_index = j
            assignments[i] = min_index
        for i in range(k):
            nums = features[np.where(assignments == i)]
            new_center = np.zeros(shape=(D,))
            for j, item in enumerate(nums):
                new_center = (new_center + item)
            new_centers[i] = new_center / len(nums)
        if np.equal(new_centers, centers).all():
            break
        centers = new_centers
        pass
        # END YOUR CODE

    return assignments


def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        # YOUR CODE HERE
        new_centers = np.zeros(shape=(k, D))
        dist = cdist(features, centers, metric='euclidean')
        assignments = np.argmin(dist, axis=1)
        for i in range(k):
            idxs = np.where(assignments == i)
            new_centers[i, :] = np.mean(features[idxs], axis=0)
        if np.equal(new_centers, centers).all():
            break
        centers = new_centers
        pass
        # END YOUR CODE

    return assignments


def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Hints
    - You may find pdist (imported from scipy.spatial.distance) useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        # YOUR CODE HERE
        distances = squareform(pdist(centers))

        for i in range(distances.shape[0]):
            distances[i, i] = 2147483647

        center_1_index, center_2_index = np.unravel_index(
            distances.argmin(), distances.shape)

        center_1 = min(center_1_index, center_2_index)
        center_2 = max(center_1_index, center_2_index)

        merged_indecies = np.where(
            (assignments == center_1) | (assignments == center_2))

        merged_values = features[merged_indecies]
        length, dim = merged_values.shape
        centroid = np.array(
            [np.sum(merged_values[:, i]) / length for i in range(dim)])
        centers[center_1] = centroid
        centers = np.delete(centers, center_2, axis=0)
        assignments[merged_indecies] = center_1
        for i, item in enumerate(assignments):
            if (item > center_2):
                assignments[i] = assignments[i] - 1
        n_clusters -= 1
        pass
        # END YOUR CODE

    return assignments


# Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H * W, C))

    # YOUR CODE HERE
    features = img.reshape((H * W, C))
    pass
    # END YOUR CODE

    return features


def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H * W, C + 2))

    # YOUR CODE HERE
    grid = (np.mgrid[0:H, 0:W])
    coords = (np.dstack((grid[0], grid[1], color))).astype('float64')

    features = coords.reshape([H * W, C + 2])
    features = features - np.mean(features, axis=0)
    features = features / np.std(features, axis=0)

    pass
    # END YOUR CODE

    return features

# Quantitative Evaluation


def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    # YOUR CODE HERE
    accuracy = np.count_nonzero(
        np.equal(mask_gt, mask)) / (mask_gt.shape[0] * mask_gt.shape[1])
    pass
    # END YOUR CODE

    return accuracy


def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
