"""
CS131 - Computer Vision: Foundations and Applications
Assignment 3
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/27/2018
Python Version: 3.5+
"""

import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = size // 2
    for i in range(size):
        for j in range(size):
            kernel[i][j] = (1/(2*np.pi*np.square(sigma)))*np.exp((-1)*(np.square(i-k)+np.square(j-k))/(2*np.square(sigma)))
    
    ### END YOUR CODE
    return kernel

def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above.

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H = img.shape[0]
    W = img.shape[1]
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    Ix2 = np.square(dx)
    Iy2 = np.square(dy)
    Ixy = dx * dy   

    mIx2 = convolve(Ix2, window, mode='constant', cval=0.0)
    mIy2 = convolve(Iy2, window, mode='constant', cval=0.0)
    mIxy = convolve(Ixy, window, mode='constant', cval=0.0)

    window = gaussian_kernel(window_size, 1.4)

    for i in range(H):
        for j in range(W):
            M = np.array([[mIx2[i, j], mIxy[i, j]], [mIxy[i, j], mIy2[i, j]]])
            response[i, j] = np.linalg.det(M) - k * np.square(np.trace(M))
        
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        If a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    feature = []
    ### YOUR CODE HERE
    pm = np.mean(patch)
    pstd = np.std(patch, ddof=1)
    if pstd == 0:
        pstd = 1
    npatch = (patch - pm)/pstd
    feature = list(npatch.flatten())
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    sdists = np.sort(dists, axis=1)
    for i in range(N):
        if sdists[i, 0]/sdists[i,1] < threshold:
            matches.append([i, np.argmin(dists[i])])
    matches = np.asarray(matches)
    ### END YOUR CODE

    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)

    Return:
        H: a matrix of shape (P, P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    ### YOUR CODE HERE
    result  = np.linalg.lstsq(p2, p1)
    H = result[0]
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE  在调用linalg.lstsq的顺序上出错
    for iter in range(n_iters):
        num_inliers = 0;
        inlies = np.zeros(N)
        sample_index = np.random.choice(N, n_samples, replace=False)  #random.randint可能会产生重复的值，random.choice参数replace=False确保了返回值不重复
        H  = np.linalg.lstsq(matched2[sample_index], matched1[sample_index])[0]
        H[:,2] = np.array([0, 0, 1])
        inlies = np.linalg.norm(matched1-np.matmul(matched2, H), ord=2, axis=1)**2 < threshold
        num_inliers = np.sum(inlies)
        if num_inliers > n_inliers:
            n_inliers = num_inliers
            max_inliers = np.array(inlies).copy()
    H  = np.linalg.lstsq(matched2[max_inliers], matched1[max_inliers])[0]
    H[:,2] = np.array([0, 0, 1])
    ### END YOUR CODE
    
    return H, orig_matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. Compute the gradient image in x and y directions (already done for you)
    2. Compute gradient histograms for each cell
    3. Flatten block of histograms into a 1D feature vector
        Here, we treat the entire patch of histograms as our block
    4. Normalize flattened block
        Normalization makes the descriptor more robust to lighting variations

    Args:
        patch: grayscale image patch of shape (H, W)
        pixels_per_cell: size of a cell with shape (M, N)

    Returns:
        block: 1D patch descriptor array of shape ((H*W*n_bins)/(M*N))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    # For each cell, keep track of gradient histrogram of size n_bins
    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    for i in range(rows):
        for j in range(cols):
            for ii in range(pixels_per_cell[0]):
                for jj in range(pixels_per_cell[1]):
                    index =  int(theta_cells[i, j, ii, jj] // degrees_per_bin)
                    if index == 9:
                        index = 0
                    cells[i, j, index] += G_cells[i, j, ii, jj]                    
    flatten_cells = cells.flatten()
    mean = np.mean(flatten_cells)
    std = np.std(flatten_cells, ddof=1)
    if std == 0:
        std = 1
    block = (flatten_cells-mean)/std
    return block


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image

    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]

    ### YOUR CODE HERE
    img1_mask = img1_mask * 1.0
    img2_mask = img2_mask * 1.0
    img1_mask[0:out_H, left_margin:right_margin+1] = np.tile(np.linspace(1.0, 0.0, num=right_margin-left_margin+1,endpoint=True), (out_H ,1) )
    img2_mask[0:out_H, left_margin:right_margin+1] = np.tile(np.linspace(0.0, 1.0, num=right_margin-left_margin+1,endpoint=True), (out_H, 1) )
    
    warped = img1_warped + img2_warped
    overlap =img1_mask*1.0 + img2_mask
    merged = warped/np.maximum(overlap, 1)

    merged = img1_warped*img1_mask + img2_warped*img2_mask
    ### END YOUR CODE

    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    # Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)

    ### YOUR CODE HERE
    print("imgs num:", len(imgs))
    Hs = []
    for i in range(len(imgs)-1):
        H, robust_matches = ransac(keypoints[i], keypoints[i+1], matches[i], threshold=1)
        Hs.append(H)
    print("H num:", len(Hs))
    print("H shape:", Hs[0].shape)
    Res = []
    images = []  #without middle img
    mid = len(imgs)//2
    for i in range(len(imgs)-1):
        tmp = np.eye(Hs[0].shape[0], Hs[0].shape[1])
        if i < mid:
            for j in range(i,mid):
                tmp = np.matmul(tmp, np.linalg.inv(Hs[j]))
        elif i>=mid:
            for j in range(mid, i+1):
                tmp = np.matmul(tmp, Hs[j])
        Res.append(tmp)
        # print(tmp)
        if i != mid:
            images.append(imgs[i])
    images.append(imgs[-1])
    print("Res shape", Res[0].shape)

    output_shape, offset = get_output_space(imgs[mid], images, Res)

    img_warpeds = []
    img_masks = []
    for i in range(len(imgs)):
        # print(i)
        if i == mid:
            img_warp = warp_image(imgs[i], np.eye(3), output_shape, offset)
            img_mask = (img_warp != -1)
            img_warp[~img_mask] = 0

            img_warpeds.append(img_warp)
            img_masks.append(img_mask)
        else:
            if i>mid:
                img_warp = warp_image(imgs[i], Res[i-1], output_shape, offset)
            else: 
                img_warp = warp_image(imgs[i], Res[i], output_shape, offset)
            img_mask = (img_warp != -1)
            img_warp[~img_mask] = 0

            img_warpeds.append(img_warp)
            img_masks.append(img_mask)

    merged = img_warpeds[0]
    overlap = img_masks[0]*1.0
    for i in range(1, len(img_warpeds)):
        merged += img_warpeds[i]
        overlap += img_masks[i]
    panorama = merged / np.maximum(overlap, 1)
    ### END YOUR CODE

    return panorama
