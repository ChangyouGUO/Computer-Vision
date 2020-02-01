"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for m in range(Hi):
        for n in range(Wi):
            sum = 0
            for i in range(Hk):
                for j in range(Wk):
                    if m-(i-Hk//2) < 0 or m-(i-Hk//2) >= Hi or n-(j-Wk//2) < 0 or n-(j-Wk//2) >=Wi:
                        sum += 0
                    else:
                        sum += kernel[i, j]*image[m-(i-Hk//2), n-(j-Wk//2)]
            out[m, n] = sum
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    for i in range(pad_height, H+pad_height):
        for j in range(pad_width, W+pad_width):
            out[i, j] = image[i-pad_height, j-pad_width]
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
   
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
     
    
    kernel_flip = np.flip(kernel, 0)
    kernel_flip = np.flip(kernel_flip, 1)
    
    image_pad = zero_pad(image, Hk//2, Wk//2)

    for i in range(Hk//2 , Hi+Hk//2):
        for j in range(Wk//2 , Wi+Wk//2):
            if Hk%2 == 0:
                out[i-Hk//2, j-Wk//2] = np.sum(kernel_flip * image_pad[i-Hk//2:i+Hk//2, j-Wk//2:j+Wk//2+1])
            else:
                out[i-Hk//2, j-Wk//2] = np.sum(kernel_flip * image_pad[i-Hk//2:i+Hk//2+1, j-Wk//2:j+Wk//2+1])
            

    ### YOUR CODE HERE
    #pass
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    kernel_fliped = np.flip(kernel, 0)
    kernel_filped = np.flip(kernel_fliped, 1)
    
    image_pad = zero_pad(image, Hk//2, Wk//2)
    
    v_kernel = kernel_filped.reshape(Hk*Wk, 1)
    v_image = np.zeros((Hi*Wi, Hk*Wk))
    for i in range(Hi):
        for j in range(Wi):
            v_image[i*Wi+j, :] = image_pad[i:i+Hk, j:j+Wk].reshape(1 , Hk*Wk)
            
    

    ### YOUR CODE HERE
    out = np.dot(v_image, v_kernel).reshape(Hi, Wi)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    
    g_fliped = np.flip(g, 0)
    g_fliped = np.flip(g_fliped, 1)
    ### YOUR CODE HERE
    out = conv_fast(f, g_fliped)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_mean = g-g.mean()
    out = cross_correlation(f, g_mean)
        
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_mean = np.mean(g)
    g_sigma = np.linalg.norm(g)
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    
    out = np.zeros((Hf, Wf))
    f_expend = zero_pad(f, Hg//2, Wg//2)
    for i in range(Hg//2, Hg//2+Hf):
        for j in range(Wg//2, Wg//2+Wf):
            if Hg%2 == 0:
                f_mean = np.mean(f_expend[i-Hg//2 : i+Hg//2, j-Wg//2 : j+Wg//2+1])
                f_sigma = np.linalg.norm(f_expend[i-Hg//2 : i+Hg//2, j-Wg//2 : j+Wg//2+1])
                out[i-Hg//2, j-Wg//2] = np.sum(((f_expend[i-Hg//2 : i+Hg//2 , j-Wg//2 : j+Wg//2+1]-f_mean)/f_sigma)*((g-g_mean)/g_sigma))
            else :
                print("Hg is not even")
    ### END YOUR CODE

    return out
