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
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphKerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

#include <cmath>
#include <iostream>
#include <typeinfo>

namespace {

// Helper Functions Start

template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> spatial_coords(
    const DataType& used_for_size) noexcept {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.;
  get<1>(x) = 2.;
  get<2>(x) = 3.;
  return x;
}

// const tnsr::i<DataVector, 3, Frame::Inertial> give_values() noexcept {
//   tnsr::i<DataVector, 3, Frame::Inertial> empty{3, 0.0};

//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       if (i == j) {
//         empty.get(i)[j] = 2.0;
//       }
//     }
//   }
//   return empty;
// }

// const std::array<double, 3> make_spin_a() {
//   const std::array<double, 3> spin_a{{0.0, 0.0, 3.0}};
//   return spin_a;
// }

// double square_a(const std::array<double, 3>& spin_a) {
//   // DataVector empty_dv(3, 0.0);
//   double a_squared = 0.0;
//   for (size_t i = 0; i < 3; ++i) {
//     a_squared += spin_a[i] * spin_a[i];
//   }
//   return a_squared;
// }

// template <typename DataType>
// const DataType make_rho(const DataType r_squared,
//                              const DataType a_squared) noexcept {
//   DataType empty(3, 0.0);
//   for (size_t i = 0; i < 3; ++i) {
//     empty[i] = sqrt(r_squared[i] + a_squared[i]);
//   }
//   return empty;
// }

// r, r_squared are Scalar<DataType> (rank 0 Tensor)
// spin_a is std::array
// template <typename DataType>
// const DataType make_rho(const Scalar<DataType>& r,
//                         const std::array<double, 3>& spin_a) noexcept {
//   const auto r_squared = get(r) * get(r);
//   const double a_squared = square_a(spin_a);
//   return sqrt((get(r) * get(r)) + a_squared);
// }

// template <typename DataType>
// const DataVector test_a_dot_x(
//     const DataType spin_a,
//     const tnsr::i<DataVector, 3, Frame::Inertial> x_coords) noexcept {
//   DataVector empty(3, 0.0);

//   for (size_t i = 0; i < 3; ++i) {
//     auto x_coords_vector = x_coords[i];
//     empty[i] = std::inner_product(spin_a.begin(), spin_a.end(),
//                                   x_coords_vector.begin(), 0.);
//   }
//   return empty;
// }

// template <typename DataType>
// const tnsr::Ij<DataVector, 3, Frame::Inertial> make_matrix_F(
//     const DataType r, const DataType rho, const DataType spin_a,
//     const DataType a_squared) {
//   tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_F{3, 0.0};

//   // Setup
//   for (int i = 0; i < 3; ++i) {
//     for (int j = 0; j < 3; ++j) {
//       matrix_F.get(i, j) = -1. / rho / cube(r);
//     }
//   }

//   // F Matrix
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       if (i == j) {
//         matrix_F.get(i, j) *= (a_squared - spin_a[i] * spin_a[j]);
//       } else {
//         matrix_F.get(i, j) *= -spin_a[i] * spin_a[j];
//       }
//     }
//   }
//   return matrix_F;
// }

// template <typename DataType>
// const tnsr::Ij<DataType, 3, Frame::Inertial> make_matrix_P(
//     const DataType r, const DataType rho, const DataType spin_a) {
//   tnsr::Ij<DataVector, 3, Frame::Inertial> matrix_P{3, 0.0};

//   // Setup
//   for (int i = 0; i < 3; ++i) {
//     for (int j = 0; j < 3; ++j) {
//       matrix_P.get(i, j) = -1. / (rho + r) / r;
//     }
//   }

//   // P matrix
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       if (i == j) {
//         matrix_P.get(i, j) *= spin_a[i] * spin_a[j];
//         matrix_P.get(i, j) += rho / r;
//       } else {
//         matrix_P.get(i, j) *= spin_a[i] * spin_a[j];
//       }
//     }
//   }
//   return matrix_P;
// }

// template <typename DataType>
// const tnsr::Ij<DataType, 3, Frame::Inertial> make_jacobian(
//     const tnsr::i<DataType, 3, Frame::Inertial> x_coords,
//     tnsr::Ij<DataType, 3, Frame::Inertial> matrix_F,
//     tnsr::Ij<DataType, 3, Frame::Inertial> matrix_P) {
//   tnsr::Ij<DataType, 3, Frame::Inertial> jacobian(3, 0.0);

//   // Jacobian
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       jacobian.get(i, j) = matrix_P.get(i, j);
//       for (size_t k = 0; k < 3; ++k) {
//         jacobian.get(i, j) +=
//             matrix_F.get(i, j) * x_coords.get(k) * x_coords.get(j);
//       }
//     }
//   }
//   return jacobian;
// }

// Helper Functions end

// Function to test all the functions
template <typename Frame, typename DataType>
void test_sph_kerr_schild(const DataType& used_for_size) noexcept {
  // Parameters for SphKerrSchild solution that become globally available
  const double mass = 1.01;
  const std::array<double, 3> spin{{0.0, 0.0, .5}};
  const std::array<double, 3> center{{1.0, 1.0, 1.0}};

  // Evaluate solution, instantiating a spherical kerrschild and calling
  // constructor
  gr::Solutions::SphKerrSchild solution(mass, spin, center);

  const auto x = spatial_coords<Frame>(used_for_size);
  std::cout << "this is spatial_coords: "
            << "\n"
            << x << "\n";

  // const auto x = make_with_value<tnsr::I<DataType, 3,
  // Frame>>(used_for_size, 1.0);
  const double null_vector_0 = -1.0;

  // Instantiated an intermediate computer and cache object
  gr::Solutions::SphKerrSchild::IntermediateComputer sks_computer(
      solution, x, null_vector_0);
  gr::Solutions::SphKerrSchild::IntermediateVars<DataType, Frame> cache(
      solution, x);

  // TESTING GROUND FOR TYPES AND SIZES

  // const auto spin_a = make_spin_a();
  // std::cout << "spin a = " << spin_a << "\n";
  // const auto r_test = make_with_value<Scalar<DataType>>(used_for_size, 1.0);
  // std::cout << "this is r_test " << r_test << "\n";

  // tnsr::i<DataType, 3, Frame> matrix_test{3, 9.};
  // std::cout << "this is matrix_test " << "\n" << matrix_test << "\n";
  // const auto rho = make_rho(r, spin_a);

  // tnsr::ijk<DataType, 3, Frame> blah{4, 0.};
  // std::cout << "This is blah: " << "\n" << blah << "\n";

  // TEST END

  // call the void functions with the sks_computer object

  // Test x_sph_minus_center
  auto x_sph_minus_center = spatial_coords<Frame>(used_for_size);
  sks_computer(
      make_not_null(&x_sph_minus_center), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::x_sph_minus_center<DataType,
                                                                      Frame>{});

  // Test r_squared
  Scalar<DataType> r_squared(3, 0.);
  sks_computer(
      make_not_null(&r_squared), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::r_squared<DataType>{});

  // Test r
  Scalar<DataType> r(3, 0.);
  sks_computer(make_not_null(&r), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::r<DataType>{});

  // Test rho
  Scalar<DataType> rho(3, 0.);
  sks_computer(make_not_null(&rho), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::rho<DataType>{});

  // matrix_F test
  tnsr::Ij<DataType, 3, Frame> matrix_F{1, 0.};
  // std::cout << "this is matrix_F:" << "\n" << matrix_F << "\n";
  sks_computer(
      make_not_null(&matrix_F), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_F<DataType,
      Frame>{});

  // matrix_P test
  tnsr::Ij<DataType, 3, Frame> matrix_P{1, 0.};
  // std::cout << "this is matrix_P:" << "\n" << matrix_P << "\n";
  sks_computer(
      make_not_null(&matrix_P), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_P<DataType,
      Frame>{});

  // jacobian Test
  tnsr::Ij<DataType, 3, Frame> jacobian{1, 0.};
  sks_computer(
      make_not_null(&jacobian), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::jacobian<DataType,
      Frame>{});

  // matrix_D test
  tnsr::Ij<DataType, 3, Frame> matrix_D{1, 0.};
  sks_computer(
      make_not_null(&matrix_D), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_D<DataType,
      Frame>{});

  // matrix_C test
  tnsr::Ij<DataType, 3, Frame> matrix_C{1, 0.};
  sks_computer(
      make_not_null(&matrix_C), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_C<DataType,
      Frame>{});

  // deriv_jacobian test
  tnsr::ijK<DataType, 3, Frame> deriv_jacobian{1, 0.};
  sks_computer(
      make_not_null(&deriv_jacobian), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::deriv_jacobian<DataType,
  Frame>{});

  // matrix_Q test
  tnsr::Ij<DataType, 3, Frame> matrix_Q{1, 0.};
  sks_computer(
      make_not_null(&matrix_Q), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::matrix_Q<DataType,
      Frame>{});

  // Check matrix_Q with analytic solution
  auto expected_matrix_Q =
      make_with_value<tnsr::Ij<DataVector, 3, Frame>>(x, 0.);
  expected_matrix_Q.get(2, 2) =
      0.255025 /
      ((get_element(r, 0) + get_element(rho, 0)) * get_element(rho, 0));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // Symmetry
      if (i == j) {
        expected_matrix_Q.get(i, j) += get_element(r, 0) / get_element(rho, 0);
      }
    }
  }
  CHECK_ITERABLE_APPROX(matrix_Q, expected_matrix_Q);

  // matrix_G1 test
  tnsr::Ij<DataType, 3, Frame> matrix_G1{1, 0.};
  sks_computer(make_not_null(&matrix_G1), make_not_null(&cache),
               gr::Solutions::SphKerrSchild::internal_tags::matrix_G1<DataType,
                                                                      Frame>{});

  // check matrix_G1 with analytic solution
  auto expected_matrix_G1 =
      make_with_value<tnsr::Ij<DataVector, 3, Frame>>(matrix_G1, 0.);
  expected_matrix_G1.get(1, 1) =
      1 / ((get_element(rho, 0) * get_element(rho, 0)) * get_element(r, 0));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        expected_matrix_G1.get(i, j) *=
            0.255025 - get_element(spin, 1) * get_element(spin, 1);
      } else {
        expected_matrix_G1.get(i, j) *=
            -get_element(spin, 1) * get_element(spin, 1);
      }
    }
  }
  CHECK_ITERABLE_APPROX(matrix_G1, expected_matrix_G1);

  // a_dot_x test
  Scalar<DataType> a_dot_x(3, 0.);
  sks_computer(
      make_not_null(&a_dot_x), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::a_dot_x<DataType>{});

  // s_number test
  Scalar<DataType> s_number(1, 0.);
  sks_computer(
      make_not_null(&s_number), make_not_null(&cache),
      gr::Solutions::SphKerrSchild::internal_tags::s_number<DataType>{});

  // Check s_number with analytic solution
  auto expected_s_number =
      make_with_value<Scalar<DataVector>>(s_number, 5.20402);
  CHECK_ITERABLE_APPROX(s_number, expected_s_number);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.SphKerrSchild",
                  "[PointwiseFunctions][Unit]") {
  // auto x_coords = give_values();
  // auto spin = make_spin_a();

  // std::cout << "this is x coords:" << x_coords << "\n";
  // std::cout << "spin = " << spin << "\n";

  // std::cout << "this is spin_a:" << spin_a << "\n";
  // std::cout << "this is a_squared:" << a_squared << "\n";
  // std::cout << "this is rho:" << make_rho(r, spin_a) << "\n";

  // std::cout << "this is a_dot_x:" << test_a_dot_x(spin_a, x_coords) << "\n";

  // auto rho = make_rho(r_squared, a_squared);

  // std::cout << "this is matrix F: "
  //           << "\n"
  //           << make_matrix_F(r, rho, spin_a, a_squared) << "\n";
  // std::cout << "this is matrix P: "
  //           << "\n"
  //           << make_matrix_P(r, rho, spin_a) << "\n";

  // auto matrix_F = make_matrix_F(r, rho, spin_a, a_squared);
  // auto matrix_P = make_matrix_P(r, rho, spin_a);

  // std::cout << "this is the jacobian:"
  //           << "\n"
  //           << make_jacobian(x_coords, matrix_F, matrix_P) << "\n";

  // SphKerrSchild::IntermediateComputer<DataType, Frame>
  // SphKerrSchild::IntermediateComputer<DataType, Frame> ks_computer(3, 0.0);
  // ks_computer(r, cache, internal_tags::r<DataType>{});

  // test_sph_kerr_schild<Frame::Inertial>(0.0);
  test_sph_kerr_schild<Frame::Inertial>(DataVector(3, 0.0));

  // TESTING GROUND FOR TYPES AND SIZES - outside of a function //

  // tensor of dv
  // tnsr::Ij<DataVector, 3, Frame::Inertial> blah_dv{3, 0.};
  // tensor of doubles
  // tnsr::i<double, 2, Frame::Inertial> blah(9.);
  // std::cout << "This is blah_dv: " << "\n" << blah_dv << "\n";

  // Test End //
}
