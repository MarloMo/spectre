// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
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
 * Fits a set of N linear functions, where N is the size of the returned
 * DataVector and the size of each DataVector in the y_values.
 * The x_values contain the set of values possibly approaching zero while
 * the y_values contain the DataVectors that contain for example the set of
 * points on a Strahlkorper. Each element of y_values is a
 * DataVector of size N. The first index to y_values selects a Strahlkorper
 * and the second index selects a specfic point on that surface.
 * The y_values are indexed by [DataVector][point_on_strahlkorper].
 * The x_values and y_values must be of the same size and type.
 * Each fit determines the zero-crossing for one of the sets of points
 * in the y_values.
 */
DataVector predicted_zero_crossing_value(
    const std::deque<double>& x_values, const std::deque<DataVector>& y_values);

}  // namespace intrp
