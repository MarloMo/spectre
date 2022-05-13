// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/DataVector.hpp"

namespace intrp {

/*!
 * \brief Predicts the zero crossing of a function.
 *
 * Fits a linear function to a set of y_values at different x_values
 * and uses the fit to predict what x_value the y_value zero will be crossed.
 */
double predicted_zero_crossing_value(const std::vector<double>& x_values,
                                     const std::vector<double>& y_values);

/*!
 * \brief Predicts the zero crossing of multiple functions contained in a vector
 * of datavectors.
 *
 * Fits a linear function to a set of datavectors for different x_values
 * and uses the fits to predict what x_values, when the y_values are zero, will
 * be crossed for each function contained in the datavector.
 */
DataVector predicted_zero_crossing_value(
    const DataVector& x_values, const std::vector<DataVector>& y_values);

}  // namespace intrp
