// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/EnergyDensity.hpp"

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarWave {
template <size_t SpatialDim>
void energy_density(
    const gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, SpatialDim, Frame::Inertial>& phi) noexcept {
  get(*result) +=
      0.5 * (get(pi) * get(pi) + get(magnitude(phi)) * get(magnitude(phi)));
}
}  // namespace ScalarWave
