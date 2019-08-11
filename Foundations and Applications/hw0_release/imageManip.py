import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    #pass
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    out = image.copy()
    ### YOUR CODE HERE
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                out[i, j, k] = 0.5*(image[i, j, k]*image[i, j, k])
    ### END YOUR CODE
    return out

def Brighthen_image(image):
    out = None
    out = image.copy()
    ### YOUR CODE HERE
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                out[i, j, k] = pow(image[i, j, k], 0.5)
    ### END YOUR CODE
    return out
    


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    out = color.rgb2gray(image)
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    out = image.copy()
    ### YOUR CODE HERE
    ### deep copy 
    ### R:G:B = 0.2 : 0.7 : 0.07
    if channel == "R":
        out[:, :, 0] = 0
    elif channel == "G":
        out[:, :, 1] = 0
    elif channel == "B":
        out[:, :, 2] = 0
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    if channel == "L":
        out = lab[:, :, 0]
    elif channel == "A":
        out = lab[:, :, 1]
    elif channel == "B":
        out = lab[:, :, 2]
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    if channel == "H":
        out = hsv[:, :, 0]
    elif channel == "S":
        out = hsv[:, :, 1]
    elif channel == "V":
        out = hsv[:, :, 2]
    ### END YOUR CODE
    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    out = image1.copy()
    
    im1 = image1.copy()
    im2 = image2.copy()
    if channel1 == "R":
        im1[:, :, 0] = 0
    elif channel1 == "G":
        im1[:, :, 1] = 0
    elif channel1 == "B":
        im1[:, :, 2] = 0
    
    if channel2 == "R":
        im2[:, :, 0] = 0
    elif channel2 == "G":
        im2[:, :, 1] = 0
    elif channel2 == "B":
        im2[:, :, 2] = 0
    ### YOUR CODE HERE
    
    ### END YOUR CODE
    
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for k in range(out.shape[2]):
                if(j < out.shape[1]/2):
                    out[i, j, k] = im1[i, j, k]
                else:
                    out[i, j, k] = im2[i, j, k]
                    
    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None
    out = image.copy()
    ### YOUR CODE HERE
    h = out.shape[0]//2
    w = out.shape[1]//2

    out[0:h, 0:w, :] = rgb_exclusion(image[0:h, 0:w,:], "R")
    out[0:h, w:-1, :] = dim_image(image[0:h, w:-1, :])
    out[h:-1, 0:w, :] = Brighthen_image(image[h:-1, 0:w, :])
    out[h:-1, w:-1, :] = rgb_exclusion(image[h:-1, w:-1, :], "R") 
    
    ### END YOUR CODE

    return out
