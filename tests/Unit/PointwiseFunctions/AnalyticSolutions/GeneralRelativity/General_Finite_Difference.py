# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# CHECKS THE DERIVATIVE OF FUNCTION F(x,y,z) WITH FINITE DIFFERENCE OF SOME
# SMALL PERTUBATION (dx=dy=dz) OF F(x,y,z)


def check_finite_difference_rank1(delta_pos_vectors, delta_neg_vectors, delta):

    # pass the vectors into a list
    delta_pos_vectors = delta_pos_vectors.tolist()
    delta_neg_vectors = delta_neg_vectors.tolist()
    # set lenght size
    length = len(delta_pos_vectors)
    derivative_tensor = []
    for i in range(length):
        # central difference formula
        derivative_tensor_indexed_value = (delta_neg_vectors[i] -
                                           delta_pos_vectors[i]) / (2 * delta)
        derivative_tensor.append(derivative_tensor_indexed_value)
    derivative_tensor = np.array(derivative_tensor)
    return derivative_tensor


def finite_difference_rank2(delta_pos_vectors, delta_neg_vectors, delta):
    '''
    Parameters
    ----------
    delta_pos_vectors: np.array of size(# of param of f): The
    function of interest evaluated at the coordinates of interest.
    Can also be one of the vectors that make up a higher
    dimensional tensor.

    delta_neg_vectors: rank 2 Tensor expressed as a np.array of
    dim(# of param of f)**2: The function of interest evaluated at the
    coordinates of interest plus some small pertubation. Need a vector
    for a pertubation in each direction, so it is a rank two tensor.
    Can also be a tensor composed of vectors of a small pertubation that
    make up a higher dimensional perturbed tensor.

    delta: np.array of size(# of param of f): Size of the
    pertubation for the parameters of the function whose derivative is
    being taken (for f(x,y,z), dx=dy=dz).

    Returns
    -------
    Type:
        rank 2 tensor
    '''

    # pass the vectors into a list
    delta_pos_vectors = delta_pos_vectors.tolist()
    delta_neg_vectors = delta_neg_vectors.tolist()
    # set lenght size
    length = len(delta_pos_vectors)
    derivative_tensor = []
    for i in range(length):
        dimension_1 = []
        for j in range(length):
            # central difference formula
            derivative_tensor_indexed_value = (
                delta_pos_vectors[j][i] - delta_neg_vectors[j][i]) / (2 *
                                                                      delta)
            dimension_1.append(derivative_tensor_indexed_value)
        derivative_tensor.append(dimension_1)
    derivative_tensor = np.array(derivative_tensor)
    return derivative_tensor


def finite_difference_rank3(delta_pos_vectors, delta_neg_vectors, delta):
    '''
    Parameters
    ----------
    delta_pos_vectors: np.array of size(# of param of f): The
    function of interest evaluated at the coordinates of interest.
    Can also be one of the vectors that make up a higher
    dimensional tensor.

    delta_neg_vectors: rank 2 Tensor expressed as a np.array of
    dim(# of param of f)**2: The function of interest evaluated at the
    coordinates of interest plus some small pertubation. Need a vector
    for a pertubation in each direction, so it is a rank two tensor.
    Can also be a tensor composed of vectors of a small pertubation that
    make up a higher dimensional perturbed tensor.

    delta: np.array of size(# of param of f): Size of the
    pertubation for the parameters of the function whose derivative is
    being taken (for f(x,y,z), dx=dy=dz).

    Returns
    -------
    Type:
        rank 3 tensor
    '''

    # pass the vectors into a list
    delta_pos_vectors = delta_pos_vectors.tolist()
    delta_neg_vectors = delta_neg_vectors.tolist()
    # set lenght size
    length = len(delta_pos_vectors)
    derivative_tensor = []
    for i in range(length):
        # Each entry is the derivative of the function
        # in the ith direction
        dimension_2 = []
        for j in range(length):
            dimension_1 = []
            for k in range(length):
                # central difference formula
                derivative_tensor_indexed_value = (
                    delta_pos_vectors[i][j][k] -
                    delta_neg_vectors[i][j][k]) / (2 * delta)
                dimension_1.append(derivative_tensor_indexed_value)
            dimension_2.append(dimension_1)
        derivative_tensor.append(dimension_2)
    derivative_tensor = np.array(derivative_tensor)
    return derivative_tensor
