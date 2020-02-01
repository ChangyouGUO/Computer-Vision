import numpy as np


def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    Args:
        a: numpy array of shape (x, n)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """
    out = np.dot(a, b)
    ### YOUR CODE HERE
    #pass
    ### END YOUR CODE
    return out


def complicated_matrix_function(M, a, b):
    """Implement (a.T * b) * (M * a.T).

    (optional): Use the `dot_product(a, b)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (1, n).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """
    squeezed_a = np.squeeze(a)
    squeezed_b = np.squeeze(b)
    left = dot_product(squeezed_a, squeezed_b)
    right = dot_product(M, a.T)
    out = dot_product(left, right)
    ### YOUR CODE HERE
    #pass
    ### END YOUR CODE

    return out


def svd(M):
    """Implement Singular Value Decomposition.

    (optional): Look up `np.linalg` library online for a list of
    helper functions that you might find useful.

    Args:
        M: numpy matrix of shape (m, n)

    Returns:
        u: numpy array of shape (m, m).
        s: numpy array of shape (k).
        v: numpy array of shape (n, n).
    """
    u = None
    s = None
    v = None
    ### YOUR CODE HERE
    #pass
    ### END YOUR CODE
    u, s, v = np.linalg.svd(M, full_matrices=True)
    return u, s, v


def get_singular_values(M, k):
    """Return top n singular values of matrix.

    (optional): Use the `svd(M)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (m, n).
        k: number of singular values to output.

    Returns:
        singular_values: array of shape (k)
    """
    singular_values = None
    u = None
    s = None
    v = None
    u, s, v = svd(M)
    ### YOUR CODE HERE
    #pass
    ### END YOUR CODE
    singular_values = s[0:k]
    return singular_values


def eigen_decomp(M):
    """Implement eigenvalue decomposition.

    Args:
        matrix: numpy matrix of shape (m, n)

    Returns:
        w: numpy array of shape (m, m) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    w = None
    v = None
    w, v = np.linalg.eig(M)
    
    ### YOUR CODE HERE
    #pass
    ### END YOUR CODE
    return w, v


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values.

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m, 1)
    """
    eigenvalues = []
    eigenvectors = []
    w = None
    v = None
    w, v = eigen_decomp(M)
    ### YOUR CODE HERE
    #pass
    ### END YOUR CODE
    eigenvalues = w[0:k]
    for i in range(k):
        tmp = []
        for j in range(v.shape[0]):
            tmp.append(v[j, i])
        eigenvectors.append(tmp)
    return eigenvalues, eigenvectors
