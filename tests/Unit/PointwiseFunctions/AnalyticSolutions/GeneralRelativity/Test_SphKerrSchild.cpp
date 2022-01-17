// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <typeinfo>

// IWYU pragma: no_forward_declare Tags::deriv

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

// set up non perturbed spatial coordinates
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> spatial_coords(const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.1;
  get<1>(x) = 3.2;
  get<2>(x) = 5.3;
  return x;
}

// Coordinates of interest plus some small perturbations

// setup perturbed spatial coordinates in the pos x direction

// setup some small perturbation (dx=dy=dz)
const double delta = 1.0e-8;

template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_pos_x_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.1 + delta;
  get<1>(x) = 3.2;
  get<2>(x) = 5.3;
  return x;
}

// setup perturbed spatial coordinates in the y direction
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_pos_y_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.1;
  get<1>(x) = 3.2 + delta;
  get<2>(x) = 5.3;
  return x;
}

// setup perturbed spatial coordinates in the z direction
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_pos_z_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.1;
  get<1>(x) = 3.2;
  get<2>(x) = 5.3 + delta;
  return x;
}

// Coordinates of interest minus some small perturbations

// setup perturbed spatial coordinates in the neg x direction

template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_neg_x_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.1 - delta;
  get<1>(x) = 3.2;
  get<2>(x) = 5.3;
  return x;
}

// setup perturbed spatial coordinates in the y direction
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_neg_y_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.1;
  get<1>(x) = 3.2 - delta;
  get<2>(x) = 5.3;
  return x;
}

// setup perturbed spatial coordinates in the z direction
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_neg_z_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.1;
  get<1>(x) = 3.2;
  get<2>(x) = 5.3 - delta;
  return x;
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.SphKerrSchild",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/");

  // Parameters for SphKerrSchild solution
  // Set up DataVector with same lenghts
  const DataVector used_for_size(1);

  const size_t used_for_sizet = used_for_size.size();
  const double mass = .5;
  const std::array<double, 3> spin{{0., 0., .1}};
  const std::array<double, 3> center{{0., 0., .1}};

  // non perturbed spatial coordinates
  const auto x = spatial_coords<Frame::Inertial>(used_for_size);

  // setup some positive perturbation size
  // perturbed spatial coordinates in the positive x direction
  const auto x_perturbed_pos_x_dir =
      perturbed_pos_x_dir_spatial_coords<Frame::Inertial>(used_for_size);
  // perturbed spatial coordinates in the positive y direction
  const auto x_perturbed_pos_y_dir =
      perturbed_pos_y_dir_spatial_coords<Frame::Inertial>(used_for_size);
  // perturbed spatial coordinates in the positive z direction
  const auto x_perturbed_pos_z_dir =
      perturbed_pos_z_dir_spatial_coords<Frame::Inertial>(used_for_size);

  // setup some minus perturbation size
  // perturbed spatial coordinates in the minus x direction
  const auto x_perturbed_neg_x_dir =
      perturbed_neg_x_dir_spatial_coords<Frame::Inertial>(used_for_size);
  // perturbed spatial coordinates in the minus y direction
  const auto x_perturbed_neg_y_dir =
      perturbed_neg_y_dir_spatial_coords<Frame::Inertial>(used_for_size);
  // perturbed spatial coordinates in the minus z direction
  const auto x_perturbed_neg_z_dir =
      perturbed_neg_z_dir_spatial_coords<Frame::Inertial>(used_for_size);

  // Set up the solution, computer object, and cache object
  gr::Solutions::SphKerrSchild solution(mass, spin, center);

  // non perturbed computer
  gr::Solutions::SphKerrSchild::IntermediateComputer sks_computer(solution, x);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache(used_for_sizet);

  // setup positive perturbation computer
  // perturbed_pos_x_direction_computer
  gr::Solutions::SphKerrSchild::IntermediateComputer
      sks_computer_pos_x_dir_perturbed(solution, x_perturbed_pos_x_dir);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache_pos_x_dir_perturbed(used_for_sizet);
  // perturbed_pos_y_direction_computer
  gr::Solutions::SphKerrSchild::IntermediateComputer
      sks_computer_pos_y_dir_perturbed(solution, x_perturbed_pos_y_dir);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache_pos_y_dir_perturbed(used_for_sizet);
  // perturbed_pos_z_direction_computer
  gr::Solutions::SphKerrSchild::IntermediateComputer
      sks_computer_pos_z_dir_perturbed(solution, x_perturbed_pos_z_dir);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache_pos_z_dir_perturbed(used_for_sizet);

  // setup minus perturbation computer
  // perturbed_neg_x_direction_computer
  gr::Solutions::SphKerrSchild::IntermediateComputer
      sks_computer_neg_x_dir_perturbed(solution, x_perturbed_neg_x_dir);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache_neg_x_dir_perturbed(used_for_sizet);
  // perturbed_neg_y_direction_computer
  gr::Solutions::SphKerrSchild::IntermediateComputer
      sks_computer_neg_y_dir_perturbed(solution, x_perturbed_neg_y_dir);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache_neg_y_dir_perturbed(used_for_sizet);
  // perturbed_neg_z_direction_computer
  gr::Solutions::SphKerrSchild::IntermediateComputer
      sks_computer_neg_z_dir_perturbed(solution, x_perturbed_neg_z_dir);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataVector, Frame::Inertial>
      cache_neg_z_dir_perturbed(used_for_sizet);

  // Functions outputs and tests

  // x_sph_minus_center test - non perturbed
  auto x_sph_minus_center = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&x_sph_minus_center), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::x_sph_minus_center<
                   DataVector, Frame::Inertial>{});

  std::cout << "x_sph_minus_center: "
            << "\n"
            << x_sph_minus_center << "\n";

  // r_squared test - non perturbed
  Scalar<DataVector> r_squared(3_st, 0.);
  sks_computer(
      make_not_null(&r_squared), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::r_squared<DataVector>{});

  std::cout << "This is r_squared: "
            << "\n"
            << r_squared << "\n";

  // r test - non perturbed
  Scalar<DataVector> r(3_st, 0.);
  sks_computer(make_not_null(&r), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::r<DataVector>{});

  std::cout << "This is r: "
            << "\n"
            << r << "\n";

  // rho test - non perturbed
  Scalar<DataVector> rho(3_st, 0.);
  sks_computer(make_not_null(&rho), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::rho<DataVector>{});

  std::cout << "This is rho: "
            << "\n"
            << rho << "\n";

  // matrix_F test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_F{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_F), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_F<DataVector,
                                                            Frame::Inertial>{});

  // matrix_P test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_P{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_P), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_P<DataVector,
                                                            Frame::Inertial>{});

  // jacobian test - non perturbed
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian{1_st, 0.};
  sks_computer(
      make_not_null(&jacobian), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  //   std::cout << "This is the jacobian: " << std::setprecision(16) << "\n"
  //             << jacobian << std::endl;

  // jacobian_perturbed_pos_x_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian_perturbed_pos_x_dir{1_st,
                                                                        0.};
  sks_computer_pos_x_dir_perturbed(
      make_not_null(&jacobian_perturbed_pos_x_dir),
      make_not_null(&cache_pos_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  //   std::cout << "This is the jacobian_perturbed_pos_x_dir: "
  //             << "\n"
  //             << jacobian_perturbed_pos_x_dir << std::endl;

  // jacobian_perturbed_pos_y_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian_perturbed_pos_y_dir{1_st,
                                                                        0.};
  sks_computer_pos_y_dir_perturbed(
      make_not_null(&jacobian_perturbed_pos_y_dir),
      make_not_null(&cache_pos_y_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  //   std::cout << "This is the jacobian_perturbed_pos_y_dir: "
  //             << "\n"
  //             << jacobian_perturbed_pos_y_dir << std::endl;

  // jacobian_perturbed_pos_z_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian_perturbed_pos_z_dir{1_st,
                                                                        0.};
  sks_computer_pos_z_dir_perturbed(
      make_not_null(&jacobian_perturbed_pos_z_dir),
      make_not_null(&cache_pos_z_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  //   std::cout << "This is the jacobian_perturbed_pos_z_dir: "
  //             << "\n"
  //             << jacobian_perturbed_pos_z_dir << std::endl;

  // jacobian_perturbed_neg_x_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian_perturbed_neg_x_dir{1_st,
                                                                        0.};
  sks_computer_neg_x_dir_perturbed(
      make_not_null(&jacobian_perturbed_neg_x_dir),
      make_not_null(&cache_neg_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  //   std::cout << "This is the jacobian_perturbed_neg_x_dir: "
  //             << "\n"
  //             << jacobian_perturbed_neg_x_dir << std::endl;

  // jacobian_perturbed_neg_y_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian_perturbed_neg_y_dir{1_st,
                                                                        0.};
  sks_computer_neg_y_dir_perturbed(
      make_not_null(&jacobian_perturbed_neg_y_dir),
      make_not_null(&cache_neg_y_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  //   std::cout << "This is the jacobian_perturbed_neg_y_dir: "
  //             << "\n"
  //             << jacobian_perturbed_neg_y_dir << std::endl;

  // jacobian_perturbed_neg_z_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian_perturbed_neg_z_dir{1_st,
                                                                        0.};
  sks_computer_neg_z_dir_perturbed(
      make_not_null(&jacobian_perturbed_neg_z_dir),
      make_not_null(&cache_neg_z_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  //   std::cout << "This is the jacobian_perturbed_neg_z_dir: "
  //             << "\n"
  //             << jacobian_perturbed_neg_z_dir << std::endl;

  // matrix_D test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_D{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_D), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_D<DataVector,
                                                            Frame::Inertial>{});

  // matrix_C test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_C{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_C), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_C<DataVector,
                                                            Frame::Inertial>{});

  // deriv_jacobian test
  tnsr::iJk<DataVector, 3, Frame::Inertial> deriv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&deriv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::deriv_jacobian<
                   DataVector, Frame::Inertial>{});

  //   std::cout << "This is the deriv_jacobian: " << std::setprecision(16) <<
  //   "\n"
  //             << deriv_jacobian << std::endl;

  // matrix_Q test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_Q{1_st, 0.};
  sks_computer(
      make_not_null(&matrix_Q), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_Q<DataVector,
                                                            Frame::Inertial>{});

  // matrix_G1 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_G1{1_st, 0.};
  sks_computer(make_not_null(&matrix_G1), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_G1<
                   DataVector, Frame::Inertial>{});

  // a_dot_x test - non perturbed
  Scalar<DataVector> a_dot_x(3_st, 0.);
  sks_computer(
      make_not_null(&a_dot_x), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::a_dot_x<DataVector>{});

  std::cout << "This is a_dot_x: "
            << "\n"
            << a_dot_x << std::endl;

  // matrix_G2 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_G2{1_st, 0.};
  sks_computer(make_not_null(&matrix_G2), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_G2<
                   DataVector, Frame::Inertial>{});

  // G1_dot_x test
  tnsr::I<DataVector, 3, Frame::Inertial> G1_dot_x{3_st, 0.};
  sks_computer(
      make_not_null(&G1_dot_x), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::G1_dot_x<DataVector,
                                                            Frame::Inertial>{});

  // G2_dot_x test
  tnsr::i<DataVector, 3, Frame::Inertial> G2_dot_x{3_st, 0.};
  sks_computer(
      make_not_null(&G2_dot_x), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::G2_dot_x<DataVector,
                                                            Frame::Inertial>{});

  // inv_jacobian test - non perturbed
  tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&inv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
                   DataVector, Frame::Inertial>{});

  // inv_jacobian_pos_x_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian_pos_x_dir{1_st, 0.};
  sks_computer_pos_x_dir_perturbed(
      make_not_null(&inv_jacobian_pos_x_dir),
      make_not_null(&cache_pos_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
          DataVector, Frame::Inertial>{});

  // inv_jacobian_pos_y_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian_pos_y_dir{1_st, 0.};
  sks_computer_pos_y_dir_perturbed(
      make_not_null(&inv_jacobian_pos_y_dir),
      make_not_null(&cache_pos_y_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
          DataVector, Frame::Inertial>{});

  // inv_jacobian_pos_z_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian_pos_z_dir{1_st, 0.};
  sks_computer_pos_z_dir_perturbed(
      make_not_null(&inv_jacobian_pos_z_dir),
      make_not_null(&cache_pos_z_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
          DataVector, Frame::Inertial>{});

  // inv_jacobian_neg_x_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian_neg_x_dir{1_st, 0.};
  sks_computer_neg_x_dir_perturbed(
      make_not_null(&inv_jacobian_neg_x_dir),
      make_not_null(&cache_neg_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
          DataVector, Frame::Inertial>{});

  // inv_jacobian_neg_y_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian_neg_y_dir{1_st, 0.};
  sks_computer_neg_y_dir_perturbed(
      make_not_null(&inv_jacobian_neg_y_dir),
      make_not_null(&cache_neg_y_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
          DataVector, Frame::Inertial>{});

  // inv_jacobian_neg_z_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian_neg_z_dir{1_st, 0.};
  sks_computer_neg_z_dir_perturbed(
      make_not_null(&inv_jacobian_neg_z_dir),
      make_not_null(&cache_neg_z_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
          DataVector, Frame::Inertial>{});

  // matrix_E1 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_E1{1_st, 0.};
  sks_computer(make_not_null(&matrix_E1), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_E1<
                   DataVector, Frame::Inertial>{});

  // matrix_E2 test
  tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_E2{1_st, 0.};
  sks_computer(make_not_null(&matrix_E2), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_E2<
                   DataVector, Frame::Inertial>{});

  // deriv_inv_jacobian test - non perturbed
  tnsr::iJk<DataVector, 3, Frame::Inertial> deriv_inv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&deriv_inv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::deriv_inv_jacobian<
                   DataVector, Frame::Inertial>{});

  // H test - non perturbed
  Scalar<DataVector> H{3_st, 0.};
  sks_computer(make_not_null(&H), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::H<DataVector>{});

  std::cout << "This is H: "
            << "\n"
            << H << "\n";

  // H_pos_x_dir test
  Scalar<DataVector> H_pos_x_dir{3_st, 0.};
  sks_computer_pos_x_dir_perturbed(
      make_not_null(&H_pos_x_dir), make_not_null(&cache_pos_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::H<DataVector>{});

  // H_pos_y_dir test
  Scalar<DataVector> H_pos_y_dir{3_st, 0.};
  sks_computer_pos_y_dir_perturbed(
      make_not_null(&H_pos_y_dir), make_not_null(&cache_pos_y_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::H<DataVector>{});

  // H_pos_z_dir test
  Scalar<DataVector> H_pos_z_dir{3_st, 0.};
  sks_computer_pos_z_dir_perturbed(
      make_not_null(&H_pos_z_dir), make_not_null(&cache_pos_z_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::H<DataVector>{});

  // H_neg_x_dir test
  Scalar<DataVector> H_neg_x_dir{3_st, 0.};
  sks_computer_neg_x_dir_perturbed(
      make_not_null(&H_neg_x_dir), make_not_null(&cache_neg_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::H<DataVector>{});

  // H_neg_y_dir test
  Scalar<DataVector> H_neg_y_dir{3_st, 0.};
  sks_computer_neg_y_dir_perturbed(
      make_not_null(&H_neg_y_dir), make_not_null(&cache_neg_y_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::H<DataVector>{});

  // H_neg_z_dir test
  Scalar<DataVector> H_neg_z_dir{3_st, 0.};
  sks_computer_neg_z_dir_perturbed(
      make_not_null(&H_neg_z_dir), make_not_null(&cache_neg_z_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::H<DataVector>{});

  // x_kerr_schild test - non perturbed
  auto x_kerr_schild = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&x_kerr_schild), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
                   DataVector, Frame::Inertial>{});

  std::cout << "This is x_kerr_schild: "
            << "\n"
            << x_kerr_schild << "\n";

  // x_kerr_schild_pos_x_dir test
  auto x_kerr_schild_perturbed_pos_x_dir =
      perturbed_pos_x_dir_spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer_pos_x_dir_perturbed(
      make_not_null(&x_kerr_schild_perturbed_pos_x_dir),
      make_not_null(&cache_pos_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
          DataVector, Frame::Inertial>{});

  //   //   std::cout << "This is x_kerr_schild_pos_x_dir: "
  //   //             << "\n"
  //   //             << x_kerr_schild_perturbed_pos_x_dir << "\n";

  // x_kerr_schild_pos_y_dir test
  auto x_kerr_schild_perturbed_pos_y_dir =
      perturbed_pos_y_dir_spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer_pos_y_dir_perturbed(
      make_not_null(&x_kerr_schild_perturbed_pos_y_dir),
      make_not_null(&cache_pos_y_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
          DataVector, Frame::Inertial>{});

  //   //   std::cout << "This is x_kerr_schild_pos_y_dir: "
  //   //             << "\n"
  //   //             << x_kerr_schild_perturbed_pos_y_dir << "\n";

  // x_kerr_schild_pos_z_dir test
  auto x_kerr_schild_perturbed_pos_z_dir =
      perturbed_pos_z_dir_spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer_pos_z_dir_perturbed(
      make_not_null(&x_kerr_schild_perturbed_pos_z_dir),
      make_not_null(&cache_pos_z_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
          DataVector, Frame::Inertial>{});

  //   //   std::cout << "This is x_kerr_schild_pos_z_dir: "
  //   //             << "\n"
  //   //             << x_kerr_schild_perturbed_pos_z_dir << "\n";

  // x_kerr_schild_neg_x_dir test
  auto x_kerr_schild_perturbed_neg_x_dir =
      perturbed_neg_x_dir_spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer_neg_x_dir_perturbed(
      make_not_null(&x_kerr_schild_perturbed_neg_x_dir),
      make_not_null(&cache_neg_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
          DataVector, Frame::Inertial>{});

  // x_kerr_schild_neg_y_dir test
  auto x_kerr_schild_perturbed_neg_y_dir =
      perturbed_neg_y_dir_spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer_neg_y_dir_perturbed(
      make_not_null(&x_kerr_schild_perturbed_neg_y_dir),
      make_not_null(&cache_neg_y_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
          DataVector, Frame::Inertial>{});

  // x_kerr_schild_neg_z_dir test
  auto x_kerr_schild_perturbed_neg_z_dir =
      perturbed_neg_z_dir_spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer_neg_z_dir_perturbed(
      make_not_null(&x_kerr_schild_perturbed_neg_z_dir),
      make_not_null(&cache_neg_z_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
          DataVector, Frame::Inertial>{});

  // a_cross_x test - non perturbed
  auto a_cross_x = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&a_cross_x), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::a_cross_x<
                   DataVector, Frame::Inertial>{});

  //   std::cout << "This is a_cross_x: "
  //             << "\n"
  //             << a_cross_x << "\n";

  // kerr_schild_l test - non perturbed
  auto kerr_schild_l = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&kerr_schild_l), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::kerr_schild_l<
                   DataVector, Frame::Inertial>{});

  //   std::cout << "This is kerr_schild_l: "
  //             << "\n"
  //             << kerr_schild_l << "\n";

  // sph_kerr_schild_l_lower test - non perturbed
  tnsr::i<DataVector, 4, Frame::Inertial> sph_kerr_schild_l_lower{
      used_for_size};
  sks_computer(
      make_not_null(&sph_kerr_schild_l_lower), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::sph_kerr_schild_l_lower<
          DataVector, Frame::Inertial>{});

  std::cout << "This is sph_kerr_schild_l_lower: "
            << "\n"
            << sph_kerr_schild_l_lower << "\n";

  // sph_kerr_schild_l_upper test - non perturbed
  tnsr::I<DataVector, 4, Frame::Inertial> sph_kerr_schild_l_upper{
      used_for_size};
  sks_computer(
      make_not_null(&sph_kerr_schild_l_upper), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::sph_kerr_schild_l_upper<
          DataVector, Frame::Inertial>{});

  std::cout << "This is sph_kerr_schild_l_upper: "
            << "\n"
            << sph_kerr_schild_l_upper << "\n";

  // deriv_H test - non perturbed
  tnsr::I<DataVector, 4, Frame::Inertial> deriv_H{used_for_size};
  sks_computer(make_not_null(&deriv_H), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::deriv_H<
                   DataVector, Frame::Inertial>{});

  std::cout << "This is deriv_H: " << "\n" << deriv_H << "\n";

  // Test Jacobian by General_Finite_Difference.py

    // setup the vectors using the perturbed coordinates in
    // the positive direction
    auto x_kerr_delta_pos_vectors =
        make_with_value<tnsr::Ij<double, 3, Frame::Inertial>>(1_st, 0.0);
    x_kerr_delta_pos_vectors.get(0, 0) =
    x_kerr_schild_perturbed_pos_x_dir[0][0]; x_kerr_delta_pos_vectors.get(0,
    1) = x_kerr_schild_perturbed_pos_x_dir[1][0];
    x_kerr_delta_pos_vectors.get(0, 2) =
    x_kerr_schild_perturbed_pos_x_dir[2][0]; x_kerr_delta_pos_vectors.get(1,
    0) = x_kerr_schild_perturbed_pos_y_dir[0][0];
    x_kerr_delta_pos_vectors.get(1, 1) =
    x_kerr_schild_perturbed_pos_y_dir[1][0]; x_kerr_delta_pos_vectors.get(1,
    2) = x_kerr_schild_perturbed_pos_y_dir[2][0];
    x_kerr_delta_pos_vectors.get(2, 0) =
    x_kerr_schild_perturbed_pos_z_dir[0][0]; x_kerr_delta_pos_vectors.get(2,
    1) = x_kerr_schild_perturbed_pos_z_dir[1][0];
    x_kerr_delta_pos_vectors.get(2, 2) =
    x_kerr_schild_perturbed_pos_z_dir[2][0];

    // setup the vectors using the perturbed coordinates in
    // the negative direction
    auto x_kerr_delta_neg_vectors =
        make_with_value<tnsr::Ij<double, 3, Frame::Inertial>>(1_st, 0.0);
    x_kerr_delta_neg_vectors.get(0, 0) =
    x_kerr_schild_perturbed_neg_x_dir[0][0]; x_kerr_delta_neg_vectors.get(0,
    1) = x_kerr_schild_perturbed_neg_x_dir[1][0];
    x_kerr_delta_neg_vectors.get(0, 2) =
    x_kerr_schild_perturbed_neg_x_dir[2][0]; x_kerr_delta_neg_vectors.get(1,
    0) = x_kerr_schild_perturbed_neg_y_dir[0][0];
    x_kerr_delta_neg_vectors.get(1, 1) =
    x_kerr_schild_perturbed_neg_y_dir[1][0]; x_kerr_delta_neg_vectors.get(1,
    2) = x_kerr_schild_perturbed_neg_y_dir[2][0];
    x_kerr_delta_neg_vectors.get(2, 0) =
    x_kerr_schild_perturbed_neg_z_dir[0][0]; x_kerr_delta_neg_vectors.get(2,
    1) = x_kerr_schild_perturbed_neg_z_dir[1][0];
    x_kerr_delta_neg_vectors.get(2, 2) =
    x_kerr_schild_perturbed_neg_z_dir[2][0];

    const auto finite_diff_jacobian =
        pypp::call<tnsr::Ij<DataVector, 3, Frame::Inertial>>(
            "General_Finite_Difference", "finite_difference_rank2",
            x_kerr_delta_pos_vectors, x_kerr_delta_neg_vectors, delta);

    Approx custom_approx = Approx::custom().epsilon(1e-6).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(jacobian, finite_diff_jacobian,
    custom_approx);

  //   Test deriv_jacobian by General_Finite_Difference.py

    // setup the vectors using the perturbed coordinates
    // in the positive direction
    auto jacobian_delta_pos_vectors =
        make_with_value<tnsr::iJk<double, 3, Frame::Inertial>>(1_st, 0.0);
    jacobian_delta_pos_vectors.get(0, 0, 0) =
    jacobian_perturbed_pos_x_dir[0][0]; jacobian_delta_pos_vectors.get(0, 1,
    0) = jacobian_perturbed_pos_x_dir[1][0];
    jacobian_delta_pos_vectors.get(0, 2, 0) =
    jacobian_perturbed_pos_x_dir[2][0]; jacobian_delta_pos_vectors.get(0, 0,
    1) = jacobian_perturbed_pos_x_dir[3][0];
    jacobian_delta_pos_vectors.get(0, 1, 1) =
    jacobian_perturbed_pos_x_dir[4][0]; jacobian_delta_pos_vectors.get(0, 2,
    1) = jacobian_perturbed_pos_x_dir[5][0];
    jacobian_delta_pos_vectors.get(0, 0, 2) =
    jacobian_perturbed_pos_x_dir[6][0]; jacobian_delta_pos_vectors.get(0, 1,
    2) = jacobian_perturbed_pos_x_dir[7][0];
    jacobian_delta_pos_vectors.get(0, 2, 2) =
    jacobian_perturbed_pos_x_dir[8][0];

    jacobian_delta_pos_vectors.get(1, 0, 0) =
    jacobian_perturbed_pos_y_dir[0][0]; jacobian_delta_pos_vectors.get(1, 1,
    0) = jacobian_perturbed_pos_y_dir[1][0];
    jacobian_delta_pos_vectors.get(1, 2, 0) =
    jacobian_perturbed_pos_y_dir[2][0]; jacobian_delta_pos_vectors.get(1, 0,
    1) = jacobian_perturbed_pos_y_dir[3][0];
    jacobian_delta_pos_vectors.get(1, 1, 1) =
    jacobian_perturbed_pos_y_dir[4][0]; jacobian_delta_pos_vectors.get(1, 2,
    1) = jacobian_perturbed_pos_y_dir[5][0];
    jacobian_delta_pos_vectors.get(1, 0, 2) =
    jacobian_perturbed_pos_y_dir[6][0]; jacobian_delta_pos_vectors.get(1, 1,
    2) = jacobian_perturbed_pos_y_dir[7][0];
    jacobian_delta_pos_vectors.get(1, 2, 2) =
    jacobian_perturbed_pos_y_dir[8][0];

    jacobian_delta_pos_vectors.get(2, 0, 0) =
    jacobian_perturbed_pos_z_dir[0][0]; jacobian_delta_pos_vectors.get(2, 1,
    0) = jacobian_perturbed_pos_z_dir[1][0];
    jacobian_delta_pos_vectors.get(2, 2, 0) =
    jacobian_perturbed_pos_z_dir[2][0]; jacobian_delta_pos_vectors.get(2, 0,
    1) = jacobian_perturbed_pos_z_dir[3][0];
    jacobian_delta_pos_vectors.get(2, 1, 1) =
    jacobian_perturbed_pos_z_dir[4][0]; jacobian_delta_pos_vectors.get(2, 2,
    1) = jacobian_perturbed_pos_z_dir[5][0];
    jacobian_delta_pos_vectors.get(2, 0, 2) =
    jacobian_perturbed_pos_z_dir[6][0]; jacobian_delta_pos_vectors.get(2, 1,
    2) = jacobian_perturbed_pos_z_dir[7][0];
    jacobian_delta_pos_vectors.get(2, 2, 2) =
    jacobian_perturbed_pos_z_dir[8][0];

    // setup the vectors using the perturbed coordinates
    // in the negative direction
    auto jacobian_delta_neg_vectors =
        make_with_value<tnsr::iJk<double, 3, Frame::Inertial>>(1_st, 0.0);
    jacobian_delta_neg_vectors.get(0, 0, 0) =
    jacobian_perturbed_neg_x_dir[0][0]; jacobian_delta_neg_vectors.get(0, 1,
    0) = jacobian_perturbed_neg_x_dir[1][0];
    jacobian_delta_neg_vectors.get(0, 2, 0) =
    jacobian_perturbed_neg_x_dir[2][0]; jacobian_delta_neg_vectors.get(0, 0,
    1) = jacobian_perturbed_neg_x_dir[3][0];
    jacobian_delta_neg_vectors.get(0, 1, 1) =
    jacobian_perturbed_neg_x_dir[4][0]; jacobian_delta_neg_vectors.get(0, 2,
    1) = jacobian_perturbed_neg_x_dir[5][0];
    jacobian_delta_neg_vectors.get(0, 0, 2) =
    jacobian_perturbed_neg_x_dir[6][0]; jacobian_delta_neg_vectors.get(0, 1,
    2) = jacobian_perturbed_neg_x_dir[7][0];
    jacobian_delta_neg_vectors.get(0, 2, 2) =
    jacobian_perturbed_neg_x_dir[8][0];

    jacobian_delta_neg_vectors.get(1, 0, 0) =
    jacobian_perturbed_neg_y_dir[0][0]; jacobian_delta_neg_vectors.get(1, 1,
    0) = jacobian_perturbed_neg_y_dir[1][0];
    jacobian_delta_neg_vectors.get(1, 2, 0) =
    jacobian_perturbed_neg_y_dir[2][0]; jacobian_delta_neg_vectors.get(1, 0,
    1) = jacobian_perturbed_neg_y_dir[3][0];
    jacobian_delta_neg_vectors.get(1, 1, 1) =
    jacobian_perturbed_neg_y_dir[4][0]; jacobian_delta_neg_vectors.get(1, 2,
    1) = jacobian_perturbed_neg_y_dir[5][0];
    jacobian_delta_neg_vectors.get(1, 0, 2) =
    jacobian_perturbed_neg_y_dir[6][0]; jacobian_delta_neg_vectors.get(1, 1,
    2) = jacobian_perturbed_neg_y_dir[7][0];
    jacobian_delta_neg_vectors.get(1, 2, 2) =
    jacobian_perturbed_neg_y_dir[8][0];

    jacobian_delta_neg_vectors.get(2, 0, 0) =
    jacobian_perturbed_neg_z_dir[0][0]; jacobian_delta_neg_vectors.get(2, 1,
    0) = jacobian_perturbed_neg_z_dir[1][0];
    jacobian_delta_neg_vectors.get(2, 2, 0) =
    jacobian_perturbed_neg_z_dir[2][0]; jacobian_delta_neg_vectors.get(2, 0,
    1) = jacobian_perturbed_neg_z_dir[3][0];
    jacobian_delta_neg_vectors.get(2, 1, 1) =
    jacobian_perturbed_neg_z_dir[4][0]; jacobian_delta_neg_vectors.get(2, 2,
    1) = jacobian_perturbed_neg_z_dir[5][0];
    jacobian_delta_neg_vectors.get(2, 0, 2) =
    jacobian_perturbed_neg_z_dir[6][0]; jacobian_delta_neg_vectors.get(2, 1,
    2) = jacobian_perturbed_neg_z_dir[7][0];
    jacobian_delta_neg_vectors.get(2, 2, 2) =
    jacobian_perturbed_neg_z_dir[8][0];

    const auto finite_diff_deriv_jacobian =
        pypp::call<tnsr::iJk<DataVector, 3, Frame::Inertial>>(
            "General_Finite_Difference", "finite_difference_rank3",
            jacobian_delta_pos_vectors, jacobian_delta_neg_vectors, delta);

    CHECK_ITERABLE_CUSTOM_APPROX(deriv_jacobian, finite_diff_deriv_jacobian,
                                 custom_approx);

  // Test deriv_inv_jacobian by General_Finite_Difference.py

    // setup the vectors using the perturbed coordinates
    // in the positive direction
    auto inv_jacobian_delta_pos_vectors =
        make_with_value<tnsr::iJk<double, 3, Frame::Inertial>>(1_st, 0.0);
    inv_jacobian_delta_pos_vectors.get(0, 0, 0) =
    inv_jacobian_pos_x_dir[0][0]; inv_jacobian_delta_pos_vectors.get(0, 1, 0)
    = inv_jacobian_pos_x_dir[1][0]; inv_jacobian_delta_pos_vectors.get(0, 2,
    0) = inv_jacobian_pos_x_dir[2][0]; inv_jacobian_delta_pos_vectors.get(0,
    0, 1) = inv_jacobian_pos_x_dir[3][0];
    inv_jacobian_delta_pos_vectors.get(0, 1, 1) =
    inv_jacobian_pos_x_dir[4][0]; inv_jacobian_delta_pos_vectors.get(0, 2, 1)
    = inv_jacobian_pos_x_dir[5][0]; inv_jacobian_delta_pos_vectors.get(0, 0,
    2) = inv_jacobian_pos_x_dir[6][0]; inv_jacobian_delta_pos_vectors.get(0,
    1, 2) = inv_jacobian_pos_x_dir[7][0];
    inv_jacobian_delta_pos_vectors.get(0, 2, 2) =
    inv_jacobian_pos_x_dir[8][0];

    inv_jacobian_delta_pos_vectors.get(1, 0, 0) =
    inv_jacobian_pos_y_dir[0][0]; inv_jacobian_delta_pos_vectors.get(1, 1, 0)
    = inv_jacobian_pos_y_dir[1][0]; inv_jacobian_delta_pos_vectors.get(1, 2,
    0) = inv_jacobian_pos_y_dir[2][0]; inv_jacobian_delta_pos_vectors.get(1,
    0, 1) = inv_jacobian_pos_y_dir[3][0];
    inv_jacobian_delta_pos_vectors.get(1, 1, 1) =
    inv_jacobian_pos_y_dir[4][0]; inv_jacobian_delta_pos_vectors.get(1, 2, 1)
    = inv_jacobian_pos_y_dir[5][0]; inv_jacobian_delta_pos_vectors.get(1, 0,
    2) = inv_jacobian_pos_y_dir[6][0]; inv_jacobian_delta_pos_vectors.get(1,
    1, 2) = inv_jacobian_pos_y_dir[7][0];
    inv_jacobian_delta_pos_vectors.get(1, 2, 2) =
    inv_jacobian_pos_y_dir[8][0];

    inv_jacobian_delta_pos_vectors.get(2, 0, 0) =
    inv_jacobian_pos_z_dir[0][0]; inv_jacobian_delta_pos_vectors.get(2, 1, 0)
    = inv_jacobian_pos_z_dir[1][0]; inv_jacobian_delta_pos_vectors.get(2, 2,
    0) = inv_jacobian_pos_z_dir[2][0]; inv_jacobian_delta_pos_vectors.get(2,
    0, 1) = inv_jacobian_pos_z_dir[3][0];
    inv_jacobian_delta_pos_vectors.get(2, 1, 1) =
    inv_jacobian_pos_z_dir[4][0]; inv_jacobian_delta_pos_vectors.get(2, 2, 1)
    = inv_jacobian_pos_z_dir[5][0]; inv_jacobian_delta_pos_vectors.get(2, 0,
    2) = inv_jacobian_pos_z_dir[6][0]; inv_jacobian_delta_pos_vectors.get(2,
    1, 2) = inv_jacobian_pos_z_dir[7][0];
    inv_jacobian_delta_pos_vectors.get(2, 2, 2) =
    inv_jacobian_pos_z_dir[8][0];

    // setup the vectors using the perturbed coordinates
    // in the negative direction
    auto inv_jacobian_delta_neg_vectors =
        make_with_value<tnsr::iJk<double, 3, Frame::Inertial>>(1_st, 0.0);
    inv_jacobian_delta_neg_vectors.get(0, 0, 0) =
    inv_jacobian_neg_x_dir[0][0]; inv_jacobian_delta_neg_vectors.get(0, 1, 0)
    = inv_jacobian_neg_x_dir[1][0]; inv_jacobian_delta_neg_vectors.get(0, 2,
    0) = inv_jacobian_neg_x_dir[2][0]; inv_jacobian_delta_neg_vectors.get(0,
    0, 1) = inv_jacobian_neg_x_dir[3][0];
    inv_jacobian_delta_neg_vectors.get(0, 1, 1) =
    inv_jacobian_neg_x_dir[4][0]; inv_jacobian_delta_neg_vectors.get(0, 2, 1)
    = inv_jacobian_neg_x_dir[5][0]; inv_jacobian_delta_neg_vectors.get(0, 0,
    2) = inv_jacobian_neg_x_dir[6][0]; inv_jacobian_delta_neg_vectors.get(0,
    1, 2) = inv_jacobian_neg_x_dir[7][0];
    inv_jacobian_delta_neg_vectors.get(0, 2, 2) =
    inv_jacobian_neg_x_dir[8][0];

    inv_jacobian_delta_neg_vectors.get(1, 0, 0) =
    inv_jacobian_neg_y_dir[0][0]; inv_jacobian_delta_neg_vectors.get(1, 1, 0)
    = inv_jacobian_neg_y_dir[1][0]; inv_jacobian_delta_neg_vectors.get(1, 2,
    0) = inv_jacobian_neg_y_dir[2][0]; inv_jacobian_delta_neg_vectors.get(1,
    0, 1) = inv_jacobian_neg_y_dir[3][0];
    inv_jacobian_delta_neg_vectors.get(1, 1, 1) =
    inv_jacobian_neg_y_dir[4][0]; inv_jacobian_delta_neg_vectors.get(1, 2, 1)
    = inv_jacobian_neg_y_dir[5][0]; inv_jacobian_delta_neg_vectors.get(1, 0,
    2) = inv_jacobian_neg_y_dir[6][0]; inv_jacobian_delta_neg_vectors.get(1,
    1, 2) = inv_jacobian_neg_y_dir[7][0];
    inv_jacobian_delta_neg_vectors.get(1, 2, 2) =
    inv_jacobian_neg_y_dir[8][0];

    inv_jacobian_delta_neg_vectors.get(2, 0, 0) =
    inv_jacobian_neg_z_dir[0][0]; inv_jacobian_delta_neg_vectors.get(2, 1, 0)
    = inv_jacobian_neg_z_dir[1][0]; inv_jacobian_delta_neg_vectors.get(2, 2,
    0) = inv_jacobian_neg_z_dir[2][0]; inv_jacobian_delta_neg_vectors.get(2,
    0, 1) = inv_jacobian_neg_z_dir[3][0];
    inv_jacobian_delta_neg_vectors.get(2, 1, 1) =
    inv_jacobian_neg_z_dir[4][0]; inv_jacobian_delta_neg_vectors.get(2, 2, 1)
    = inv_jacobian_neg_z_dir[5][0]; inv_jacobian_delta_neg_vectors.get(2, 0,
    2) = inv_jacobian_neg_z_dir[6][0]; inv_jacobian_delta_neg_vectors.get(2,
    1, 2) = inv_jacobian_neg_z_dir[7][0];
    inv_jacobian_delta_neg_vectors.get(2, 2, 2) =
    inv_jacobian_neg_z_dir[8][0];

    const auto finite_diff_deriv_inv_jacobian =
        pypp::call<tnsr::iJk<DataVector, 3, Frame::Inertial>>(
            "General_Finite_Difference", "finite_difference_rank3",
            inv_jacobian_delta_pos_vectors, inv_jacobian_delta_neg_vectors,
            delta);

    CHECK_ITERABLE_CUSTOM_APPROX(deriv_inv_jacobian,
                                 finite_diff_deriv_inv_jacobian,
                                 custom_approx);

    // Test deriv_H by General_Finite_Difference.py

    // const auto finite_diff_deriv_H =
    //     pypp::call<tnsr::Ij<DataVector, 4, Frame::Inertial>>(
    //         "General_Finite_Difference", "finite_difference_rank2",
    //         x_kerr_delta_pos_vectors, x_kerr_delta_neg_vectors, delta);

    // Approx custom_approx = Approx::custom().epsilon(1e-6).scale(1.0);
    // CHECK_ITERABLE_CUSTOM_APPROX(jacobian, finite_diff_jacobian,
    // custom_approx);
}
