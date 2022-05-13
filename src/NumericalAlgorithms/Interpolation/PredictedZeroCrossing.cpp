// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/PredictedZeroCrossing.hpp"

#include <vector>

#include "NumericalAlgorithms/Interpolation/LinearLeastSquares.hpp"

namespace intrp {

double predicted_zero_crossing_value(const std::vector<double>& x_values,
                                     const std::vector<double>& y_values) {
  intrp::LinearLeastSquares<1> predictor{x_values.size()};
  const auto coefficients = predictor.fit_coefficients(x_values, y_values);
  return -coefficients[0]/coefficients[1];
}

DataVector predicted_zero_crossing_value(
    const DataVector& x_values, const std::vector<DataVector>& y_values) {
  intrp::LinearLeastSquares<1> predictor{x_values.size()};
  DataVector result(y_values.size());
  for (size_t i = 0; i < y_values.size(); i++) {
    const auto coefficients = predictor.fit_coefficients(x_values, y_values);
    result[i] = -coefficients[i][0] / coefficients[i][1];
  }
  return result;
}

}  // namespace intrp
