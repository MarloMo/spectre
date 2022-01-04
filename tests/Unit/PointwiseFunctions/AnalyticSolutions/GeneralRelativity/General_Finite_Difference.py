# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# CHECKS THE DERIVATIVE OF FUNCTION F WITH FINITE DIFFERENCE OF SOME
# SMALL PERTUBATION OF F


def combine_pert_coords(x_plus_dx_i_KS, x_plus_dy_i_KS, x_plus_dz_i_KS):
    pertubation_coords = []
    pertubation_coords.append(x_plus_dx_i_KS)
    pertubation_coords.append(x_plus_dy_i_KS)
    pertubation_coords.append(x_plus_dz_i_KS)

    pertubation_coords = np.array(pertubation_coords)
    return pertubation_coords


def central_finite_difference(perturbed_input_vectors_neg_diff,
                              perturbed_input_vectors_pos_diff, pertubation):
    '''
    Parameters
    ----------
    input_vector: np.array of size(# of param of f): The function of interest
    evaluated at the coordinates of interest. Can also be one of
    the vectors that make up a higher dimensional tensor.

    perturbed_input_vectors: rank 2 Tensor expressed as a np.array of
    dim(# of param of f)**2: The function of interest evaluated at the
    coordinates of interest plus some small pertubation. Need a vector
    for a pertubation in each direction, so it is a rank two tensor.
    Can also be a tensor composed of vectors of a small pertubation that
    make up a higher dimensional perturbed tensor.

    pertubation: np.array of size(# of param of f): Size of the
    pertubation for the parameters of the function whose derivative is
    being taken (for f(x,y,z), dx=dy=dz).

    Returns
    -------
    Difference formula:
        central: (f(x+dx) - f(x-dx))/2dx
    '''

    # pass the vectors into a list
    perturbed_input_vectors_neg_diff = perturbed_input_vectors_neg_diff.tolist()
    perturbed_input_vectors_pos_diff = perturbed_input_vectors_pos_diff.tolist()
    pertubation = pertubation.tolist()
    derivative_tensor = []
    for i in range(len(perturbed_input_vectors_neg_diff)):
        dimension_1 = []
        for j in range(len(perturbed_input_vectors_neg_diff)):
            # central difference formula
            derivative_tensor_indexed_value = (
                perturbed_input_vectors_pos_diff[j][i] -
                perturbed_input_vectors_neg_diff[j][i]) / pertubation[i]
            dimension_1.append(derivative_tensor_indexed_value)
        derivative_tensor.append(dimension_1)
    derivative_tensor = np.array(derivative_tensor)
    return derivative_tensor


def custom_finite_difference(input_vector, perturbed_input_vectors,
                             pertubation, method):
    '''
    Parameters
    ----------
    input_vector: np.array of size(# of param of f): The function of interest
    evaluated at the coordinates of interest. Can also be one of
    the vectors that make up a higher dimensional tensor.

    perturbed_input_vectors: rank 2 Tensor expressed as a np.array of
    dim(# of param of f)**2: The function of interest evaluated at the
    coordinates of interest plus some small pertubation. Need a vector
    for a pertubation in each direction, so it is a rank two tensor.
    Can also be a tensor composed of vectors of a small pertubation that
    make up a higher dimensional perturbed tensor.

    pertubation: np.array of size(# of param of f): Size of the
    pertubation for the parameters of the function whose derivative is
    being taken (for f(x,y,z), dx=dy=dz).

    Returns
    -------
    Difference formula:
        forward: (f(x+dx) - f(x))/dx
        backward: (f(x) - f(x-dx))/dx
    '''

    # pass the vectors into a list
    input_vector = input_vector.tolist()
    perturbed_input_vectors = perturbed_input_vectors.tolist()
    pertubation = pertubation.tolist()
    # selected method
    if method == 'forward':
        derivative_tensor = []
        for i in range(len(input_vector)):
            dimension_1 = []
            for j in range(len(input_vector)):
                # forward difference formula
                derivative_tensor_indexed_value = (
                    perturbed_input_vectors[j][i] -
                    input_vector[i]) / pertubation[i]
                dimension_1.append(derivative_tensor_indexed_value)
            derivative_tensor.append(dimension_1)
        derivative_tensor = np.array(derivative_tensor)
        return derivative_tensor
    elif method == 'backward':
        derivative_tensor = []
        for i in range(len(input_vector)):
            dimension_1 = []
            for j in range(len(input_vector)):
                # backward difference formula
                derivative_tensor_indexed_value = (
                    input_vector[i] -
                    perturbed_input_vectors[j][i]) / pertubation[i]
                dimension_1.append(derivative_tensor_indexed_value)
            derivative_tensor.append(dimension_1)
        derivative_tensor = np.array(derivative_tensor)
        return derivative_tensor
    else:
        raise ValueError(
            "Method not recognized. Must be 'forward' or 'backward'."
        )
