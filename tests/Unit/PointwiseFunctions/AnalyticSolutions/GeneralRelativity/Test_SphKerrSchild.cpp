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
  return x;
}

// setup some positive perturbation size
const double perturbation_size_pos = 1.0e-6;
// setup some negative perturbation size
const double perturbation_size_neg = -1.0e-6;

// Coordinates of interest plus some small perturbations
// setup perturbed spatial coordinates in the x direction
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_pos_x_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = perturbation_size_pos;
  return x;
}

// setup perturbed spatial coordinates in the y direction
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_pos_y_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<1>(x) = perturbation_size_pos;
  return x;
}

// setup perturbed spatial coordinates in the z direction
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_pos_z_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<2>(x) = perturbation_size_pos;
  return x;
}

// Coordinates of interest minus some small perturbations
// setup perturbed spatial coordinates in the x direction
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_neg_x_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = perturbation_size_neg;
  return x;
}

// setup perturbed spatial coordinates in the y direction
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_neg_y_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<1>(x) = perturbation_size_neg;
  return x;
}

// setup perturbed spatial coordinates in the z direction
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> perturbed_neg_z_dir_spatial_coords(
    const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<2>(x) = perturbation_size_neg;
  return x;
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.SphKerrSchild",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/");

  // Parameters for SphKerrSchild solution
  // Set up DataVector with same lenghts
  const DataVector used_for_size(3);
  //   const size_t num_points_1d = 2;
  //   const size_t num_points_3d = num_points_1d * num_points_1d *
  //   num_points_1d; const DataVector used_for_size(num_points_3d);

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

  // x_sph_minus_center_perturbed_pos_x_dir test
  auto x_sph_minus_center_perturbed_pos_x_dir =
      perturbed_pos_x_dir_spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer_pos_x_dir_perturbed(
      make_not_null(&x_sph_minus_center_perturbed_pos_x_dir),
      make_not_null(&cache_pos_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::x_sph_minus_center<
          DataVector, Frame::Inertial>{});

  std::cout << "x_sph_minus_center_perturbed_pos_x_dir: "
            << "\n"
            << x_sph_minus_center_perturbed_pos_x_dir << "\n";

  // x_sph_minus_center_perturbed_neg_x_dir test
  auto x_sph_minus_center_perturbed_neg_x_dir =
      perturbed_neg_x_dir_spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer_neg_x_dir_perturbed(
      make_not_null(&x_sph_minus_center_perturbed_neg_x_dir),
      make_not_null(&cache_neg_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::x_sph_minus_center<
          DataVector, Frame::Inertial>{});

  std::cout << "x_sph_minus_center_perturbed_neg_x_dir: "
            << "\n"
            << x_sph_minus_center_perturbed_neg_x_dir << "\n";

  //   // x_sph_minus_center_perturbed_z_dir test
  //   auto x_sph_minus_center_perturbed_z_dir =
  //       perturbed_z_dir_spatial_coords<Frame::Inertial>(used_for_size);
  //   sks_computer_z_dir_perturbed(
  //       make_not_null(&x_sph_minus_center_perturbed_z_dir),
  //       make_not_null(&cache_z_dir_perturbed),
  //       gr::Solutions::SphKerrSchild::internal_tags::x_sph_minus_center<
  //           DataVector, Frame::Inertial>{});

  //   std::cout << "x_sph_minus_center_perturbed_z_dir: "
  //             << "\n"
  //             << x_sph_minus_center_perturbed_z_dir << "\n";

  // r_squared test - non perturbed
  Scalar<DataVector> r_squared(3_st, 0.);
  sks_computer(
      make_not_null(&r_squared), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::r_squared<DataVector>{});

  // r test
  Scalar<DataVector> r(3_st, 0.);
  sks_computer(make_not_null(&r), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::r<DataVector>{});

  // rho test
  Scalar<DataVector> rho(3_st, 0.);
  sks_computer(make_not_null(&rho), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::rho<DataVector>{});

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

  std::cout << "This is the jacobian: " << std::setprecision(16) << "\n"
            << jacobian << std::endl;

  // jacobian_perturbed_x_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian_perturbed_pos_x_dir{1_st,
                                                                        0.};
  sks_computer_pos_x_dir_perturbed(
      make_not_null(&jacobian_perturbed_pos_x_dir),
      make_not_null(&cache_pos_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  std::cout << "This is the jacobian_perturbed_pos_x_dir: "
            << "\n"
            << jacobian_perturbed_pos_x_dir << std::endl;

  // jacobian_perturbed_y_dir test
  tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian_perturbed_neg_x_dir{1_st,
                                                                        0.};
  sks_computer_neg_x_dir_perturbed(
      make_not_null(&jacobian_perturbed_neg_x_dir),
      make_not_null(&cache_neg_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
                                                            Frame::Inertial>{});

  std::cout << "This is the jacobian_perturbed_neg_x_dir: "
            << "\n"
            << jacobian_perturbed_neg_x_dir << std::endl;

  // jacobian_perturbed_z_dir test
  //   tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian_perturbed_z_dir{1_st,
  //   0.}; sks_computer_z_dir_perturbed(
  //       make_not_null(&jacobian_perturbed_z_dir),
  //       make_not_null(&cache_z_dir_perturbed),
  //       gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
  //                                                    Frame::Inertial>{});

  //   std::cout << "This is the jacobian_perturbed_z_dir: "
  //             << "\n"
  //             << jacobian_perturbed_z_dir << std::endl;

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

  // PARTIAL DERIVATIVES FUNCTION, WORKING BUT I THINK THIS FUNCTION IS NOT
  // THE RIGHT FUNCTION TO USE.

  //   // Setup grid
  //   const std::array<double, 3> lower_bound{{-1., -1., -1.}};
  //   const std::array<double, 3> upper_bound{{1., 1., 1.}};
  //   const size_t SpatialDim = 3;
  //   Mesh<SpatialDim> mesh{num_points_1d, Spectral::Basis::Legendre,
  //                         Spectral::Quadrature::GaussLobatto};
  //   // std::cout << "This is the mesh: " << "\n" << mesh << std::endl;

  //   const auto coord_map =
  //       domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
  //           Affine3D{
  //               Affine{-1., 1., lower_bound[0], upper_bound[0]},
  //               Affine{-1., 1., lower_bound[1], upper_bound[1]},

  //               Affine{-1., 1., lower_bound[2], upper_bound[2]},
  //           });

  // Setup coordinates
  //   const auto x_logical = logical_coordinates(mesh);
  //   std::cout << "this is x_logical: "
  //             << "\n"
  //             << x_logical << std::endl;
  //   const auto x_prime = coord_map(x_logical);
  //   std::cout << "this is x_prime: "
  //             << "\n"
  //             << x_prime << std::endl;

  // Evaluate analytic solution
  //   tnsr::Ij<DataVector, 3, Frame::Inertial> jacobian{1_st, 0.};
  //   sks_computer(
  //       make_not_null(&jacobian), make_not_null(&cache),
  //       gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
  //                                                      Frame::Inertial>{});

  // Scalar<DataVector> r_squared(3_st, 0.);
  //   sks_computer(
  //       make_not_null(&r_squared), make_not_null(&cache),
  //    gr::Solutions::SphKerrSchild::internal_tags::r_squared<DataVector>{});

  // Compute actual analytical derivative of the determinant
  //   tnsr::iJk<DataVector, 3, Frame::Inertial> deriv_jacobian{1_st, 0.};
  //   sks_computer(make_not_null(&deriv_jacobian), make_not_null(&cache),
  //                gr::Solutions::SphKerrSchild::internal_tags::deriv_jacobian<
  //                    DataVector, Frame::Inertial>{});

  // Compute expected numerical derivative of the jaccobian
  //   using r_squared_tag =
  //       gr::Solutions::SphKerrSchild::internal_tags::r_squared<DataVector>;
  //   Variables<tmpl::list<r_squared_tag>>r_squared_var(num_points_3d);
  //   get<r_squared_tag>(r_squared_var) = r_squared;
  //   const auto expected_deriv_r_squared_var =
  //       partial_derivatives<tmpl::list<r_squared_tag>>(
  //           r_squared_var, mesh, coord_map.inv_jacobian(x_logical));
  //   const auto expected_deriv_r_squared =
  //       get<Tags::deriv<r_squared_tag, tmpl::size_t<SpatialDim>,
  //       Frame::Inertial>>(
  //           expected_deriv_r_squared_var);

  //   std::cout << "this is expected_deriv_r_squared"
  //             << "\n"
  //             << expected_deriv_r_squared << std::endl;

  //   Approx custom_approx = Approx::custom().epsilon(1e-11).scale(1.0);
  //   CHECK_ITERABLE_CUSTOM_APPROX(deriv_jacobian, expected_deriv_jacobian,
  //    custom_approx);

  // CHECK_ITERABLE_APPROX(deriv_jacobian, expected_deriv_jacobian);

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

  // a_dot_x test
  Scalar<DataVector> a_dot_x(3_st, 0.);
  sks_computer(
      make_not_null(&a_dot_x), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::a_dot_x<DataVector>{});

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

  // inv_jacobian test
  tnsr::Ij<DataVector, 3, Frame::Inertial> inv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&inv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<
                   DataVector, Frame::Inertial>{});

  // inv_jacobian and jacobian product test _ NO IDEA WHAT I AM DOING
  // using jacobian_tag =
  //     gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataVector,
  //                                                         Frame::Inertial>;
  // Variables<tmpl::list<jacobian_tag>> jacobian_var(3);
  // get<jacobian_tag>(jacobian_var) = jacobian;
  // using inv_jacobian_tag =
  //     gr::Solutions::SphKerrSchild::internal_tags::inv_jacobian<DataVector,
  //                                                         Frame::Inertial>;
  // Variables<tmpl::list<inv_jacobian_tag>> inv_jacobian_var(3);
  // get<inv_jacobian_tag>(inv_jacobian_var) = inv_jacobian;

  // auto product = jacobian * inv_jacobian;

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

  // deriv_inv_jacobian test
  tnsr::ijK<DataVector, 3, Frame::Inertial> deriv_inv_jacobian{1_st, 0.};
  sks_computer(make_not_null(&deriv_inv_jacobian), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::deriv_inv_jacobian<
                   DataVector, Frame::Inertial>{});

  // x_kerr_schild test - non perturbed
  auto x_kerr_schild = spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer(make_not_null(&x_kerr_schild), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
                   DataVector, Frame::Inertial>{});

  std::cout << "This is x_kerr_schild: "
            << "\n"
            << x_kerr_schild << "\n";

  // x kerrschild in the positive direction
  // x_kerr_schild_pos_x_dir test
  auto x_kerr_schild_perturbed_pos_x_dir =
      perturbed_pos_x_dir_spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer_pos_x_dir_perturbed(
      make_not_null(&x_kerr_schild_perturbed_pos_x_dir),
      make_not_null(&cache_pos_x_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
          DataVector, Frame::Inertial>{});

  // x_kerr_schild_pos_y_dir test
  auto x_kerr_schild_perturbed_pos_y_dir =
      perturbed_pos_y_dir_spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer_pos_y_dir_perturbed(
      make_not_null(&x_kerr_schild_perturbed_pos_y_dir),
      make_not_null(&cache_pos_y_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
          DataVector, Frame::Inertial>{});

  // x_kerr_schild_pos_z_dir test
  auto x_kerr_schild_perturbed_pos_z_dir =
      perturbed_pos_z_dir_spatial_coords<Frame::Inertial>(used_for_size);
  sks_computer_pos_z_dir_perturbed(
      make_not_null(&x_kerr_schild_perturbed_pos_z_dir),
      make_not_null(&cache_pos_z_dir_perturbed),
      gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
          DataVector, Frame::Inertial>{});

  // x kerrschild in the negative direction
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

  // Test the Jacobian function by General_Finite_Difference.py

  //   // CENTRAL METHOD ATTEMPT - NOT WORKING
  //   // setup the perturbed input vector using the perturbed coordinates in
  //   // the positive direction
  //   auto x_kerr_perturbed_input_vectors_pos =
  //       make_with_value<tnsr::Ij<double, 3, Frame::Inertial>>(1_st, 0.0);
  //   x_kerr_perturbed_input_vectors_pos.get(0, 0) =
  //       x_kerr_schild_perturbed_pos_x_dir[0][0];
  //   x_kerr_perturbed_input_vectors_pos.get(0, 1) =
  //       x_kerr_schild_perturbed_pos_x_dir[1][0];
  //   x_kerr_perturbed_input_vectors_pos.get(0, 2) =
  //       x_kerr_schild_perturbed_pos_x_dir[2][0];
  //   x_kerr_perturbed_input_vectors_pos.get(1, 0) =
  //       x_kerr_schild_perturbed_pos_y_dir[0][0];
  //   x_kerr_perturbed_input_vectors_pos.get(1, 1) =
  //       x_kerr_schild_perturbed_pos_y_dir[1][0];
  //   x_kerr_perturbed_input_vectors_pos.get(1, 2) =
  //       x_kerr_schild_perturbed_pos_y_dir[2][0];
  //   x_kerr_perturbed_input_vectors_pos.get(2, 0) =
  //       x_kerr_schild_perturbed_pos_z_dir[0][0];
  //   x_kerr_perturbed_input_vectors_pos.get(2, 1) =
  //       x_kerr_schild_perturbed_pos_z_dir[1][0];
  //   x_kerr_perturbed_input_vectors_pos.get(2, 2) =
  //       x_kerr_schild_perturbed_pos_z_dir[2][0];

  //   // setup the perturbed input vector using the perturbed coordinates in
  //   // the negative direction
  //   auto x_kerr_perturbed_input_vectors_neg =
  //       make_with_value<tnsr::Ij<double, 3, Frame::Inertial>>(1_st, 0.0);
  //   x_kerr_perturbed_input_vectors_neg.get(0, 0) =
  //       x_kerr_schild_perturbed_neg_x_dir[0][0];
  //   x_kerr_perturbed_input_vectors_neg.get(0, 1) =
  //       x_kerr_schild_perturbed_neg_x_dir[1][0];
  //   x_kerr_perturbed_input_vectors_neg.get(0, 2) =
  //       x_kerr_schild_perturbed_neg_x_dir[2][0];
  //   x_kerr_perturbed_input_vectors_neg.get(1, 0) =
  //       x_kerr_schild_perturbed_neg_y_dir[0][0];
  //   x_kerr_perturbed_input_vectors_neg.get(1, 1) =
  //       x_kerr_schild_perturbed_neg_y_dir[1][0];
  //   x_kerr_perturbed_input_vectors_neg.get(1, 2) =
  //       x_kerr_schild_perturbed_neg_y_dir[2][0];
  //   x_kerr_perturbed_input_vectors_neg.get(2, 0) =
  //       x_kerr_schild_perturbed_neg_z_dir[0][0];
  //   x_kerr_perturbed_input_vectors_neg.get(2, 1) =
  //       x_kerr_schild_perturbed_neg_z_dir[1][0];
  //   x_kerr_perturbed_input_vectors_neg.get(2, 2) =
  //       x_kerr_schild_perturbed_neg_z_dir[2][0];

  //   // Call the General_Finite_Difference.py to test
  //   central_finite_difference
  //   // setup the perturbation size (must be positive)
  //   const double perturbation_size_test_case = 1.0e-6;
  //   // set up the  perturbed input vector
  //   auto pertubation = make_with_value<tnsr::I<double, 3, Frame::Inertial>>(
  //       1_st, perturbation_size_test_case);

  //   const auto finite_diff_jacobian =
  //       pypp::call<tnsr::Ij<DataVector, 3, Frame::Inertial>>(
  //           "General_Finite_Difference", "central_finite_difference",
  //           x_kerr_perturbed_input_vectors_neg,
  //           x_kerr_perturbed_input_vectors_pos, pertubation);

  // CUSTOM FINITE DIFFERENCE ATTEMPT - WORKING
  // set up the input vector using the non perturbed coordinates
  const tnsr::I<DataVector, 3, Frame::Inertial>& x_kerr_input_coords =
      cache.get_var(sks_computer,
                    gr::Solutions::SphKerrSchild::internal_tags::x_kerr_schild<
                        DataVector, Frame::Inertial>{});

  // setup the perturbed input vector using the perturbed coordinates
  auto x_kerr_perturbed_input_vectors =
      make_with_value<tnsr::Ij<double, 3, Frame::Inertial>>(1_st, 0.0);
  x_kerr_perturbed_input_vectors.get(0, 0) =
      x_kerr_schild_perturbed_pos_x_dir[0][0];
  x_kerr_perturbed_input_vectors.get(0, 1) =
      x_kerr_schild_perturbed_pos_x_dir[1][0];
  x_kerr_perturbed_input_vectors.get(0, 2) =
      x_kerr_schild_perturbed_pos_x_dir[2][0];
  x_kerr_perturbed_input_vectors.get(1, 0) =
      x_kerr_schild_perturbed_pos_y_dir[0][0];
  x_kerr_perturbed_input_vectors.get(1, 1) =
      x_kerr_schild_perturbed_pos_y_dir[1][0];
  x_kerr_perturbed_input_vectors.get(1, 2) =
      x_kerr_schild_perturbed_pos_y_dir[2][0];
  x_kerr_perturbed_input_vectors.get(2, 0) =
      x_kerr_schild_perturbed_pos_z_dir[0][0];
  x_kerr_perturbed_input_vectors.get(2, 1) =
      x_kerr_schild_perturbed_pos_z_dir[1][0];
  x_kerr_perturbed_input_vectors.get(2, 2) =
      x_kerr_schild_perturbed_pos_z_dir[2][0];

  // Call the General_Finite_Difference.py to test
  // custom_check_finite_difference
  std::string method = "forward";
  // setup the perturbation size (must be positive)
  const double perturbation_size_test_case = 1.0e-6;
  // set up the  perturbed input vector
  auto pertubation = make_with_value<tnsr::I<double, 3, Frame::Inertial>>(
      1_st, perturbation_size_test_case);

  const auto finite_diff_jacobian =
      pypp::call<tnsr::Ij<DataVector, 3, Frame::Inertial>>(
          "General_Finite_Difference", "custom_finite_difference",
          x_kerr_input_coords, x_kerr_perturbed_input_vectors, pertubation,
          method);

  // Check the analytical jacobian against the finite difference jacobian
  Approx custom_approx = Approx::custom().epsilon(1e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(jacobian, finite_diff_jacobian, custom_approx);
}
