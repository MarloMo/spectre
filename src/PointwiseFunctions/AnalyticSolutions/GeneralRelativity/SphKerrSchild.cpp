// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphKerrSchild.hpp"

#include <cmath>  // IWYU pragma: keep
#include <numeric>
#include <ostream>
#include <typeinfo>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

#include <iostream>

namespace gr::Solutions {

SphKerrSchild::SphKerrSchild(const double mass,
                             SphKerrSchild::Spin::type dimensionless_spin,
                             SphKerrSchild::Center::type center,
                             const Options::Context& context)

    : mass_(mass),
      // clang-tidy: do not std::move trivial types.
      dimensionless_spin_(std::move(dimensionless_spin)),  // NOLINT
      // clang-tidy: do not std::move trivial types.
      center_(std::move(center))  // NOLINT
{
  const double spin_magnitude = magnitude(dimensionless_spin_);
  if (spin_magnitude > 1.0) {
    PARSE_ERROR(context, "Spin magnitude must be < 1. Given spin: "
                             << dimensionless_spin_ << " with magnitude "
                             << spin_magnitude);
  }
  if (mass_ < 0.0) {
    PARSE_ERROR(context, "Mass must be non-negative. Given mass: " << mass_);
  }
}

void SphKerrSchild::pup(PUP::er& p) {
  p | mass_;
  p | dimensionless_spin_;
  p | center_;
}

template <typename DataType, typename Frame>
SphKerrSchild::IntermediateComputer<DataType, Frame>::IntermediateComputer(
    const SphKerrSchild& solution, const tnsr::I<DataType, 3, Frame>& x)
    : solution_(solution), x_(x) {}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_sph_minus_center,
    const gsl::not_null<CachedBuffer*> /*cache*/,
    internal_tags::x_sph_minus_center<DataType, Frame> /*meta*/) const {
  // Instantiations
  *x_sph_minus_center = x_;

  // x_sphere_minus_center Calculation
  for (size_t i = 0; i < 3; ++i) {
    x_sph_minus_center->get(i) -= gsl::at(solution_.center(), i);
  }

  std::cout << "this is x_sph_minus_center " << *x_sph_minus_center << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r_squared,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r_squared<DataType> /*meta*/) const {
  // Instantiations
  const auto& x_sph_minus_center = cache->get_var(
      *this, internal_tags::x_sph_minus_center<DataType, Frame>{});

  // r_squared Calculation
  r_squared->get() = square(x_sph_minus_center.get(0));
  for (size_t i = 1; i < 3; ++i) {
    r_squared->get() += square(x_sph_minus_center.get(i));
  }
  std::cout << "this is r_squared " << *r_squared << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> r,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::r<DataType> /*meta*/) const {
  // Instantiations
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));

  // r Calculation
  get(*r) = sqrt(r_squared);

  std::cout << "this is r " << *r << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<Scalar<DataType>*> rho,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::rho<DataType> /*meta*/) const {
  // Instantiations
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);

  // rho Calculation
  get(*rho) = sqrt(r_squared + a_squared);

  std::cout << "this is spin_a " << spin_a << "\n";
  std::cout << "this is a_squared " << a_squared << "\n";
  std::cout << "this is rho " << *rho << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_F,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_F<DataType, Frame> /*meta*/) const {
  // Instantiations
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  // matrix_F Calculation
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_F->get(i, j) = -1. / rho / cube(r);
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      // Kronecker delta
      if (i == j) {
        matrix_F->get(i, j) *=
            (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, j));
      } else {
        matrix_F->get(i, j) *= -gsl::at(spin_a, i) * gsl::at(spin_a, j);
      }
    }
  }
  std::cout << "this is matrix_F:"
            << "\n"
            << *matrix_F << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_P,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_P<DataType, Frame> /*meta*/) const {
  // Instantiations
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  // matrix_P Calculation
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_P->get(i, j) = -1. / (rho + r) / r;
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        matrix_P->get(i, j) *= gsl::at(spin_a, i) * gsl::at(spin_a, j);
        matrix_P->get(i, j) += rho / r;
      } else {
        matrix_P->get(i, j) *= gsl::at(spin_a, i) * gsl::at(spin_a, j);
      }
    }
  }
  std::cout << "this is matrix_P:"
            << "\n"
            << *matrix_P << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::jacobian<DataType, Frame> /*meta*/) const {
  // Instantiations
  const auto& x_sph_minus_center = cache->get_var(
      *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& matrix_P =
      cache->get_var(*this, internal_tags::matrix_P<DataType, Frame>{});
  const auto& matrix_F =
      cache->get_var(*this, internal_tags::matrix_F<DataType, Frame>{});

  // jacobian Calculation
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jacobian->get(i, j) = matrix_P.get(i, j);
      for (size_t k = 0; k < 3; ++k) {
        jacobian->get(i, j) += matrix_F.get(i, k) * x_sph_minus_center.get(k) *
                               x_sph_minus_center.get(j);
      }
    }
  }
  std::cout << "this is the jacobian:"
            << "\n"
            << *jacobian << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_D,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_D<DataType, Frame> /*meta*/) const {
  // Instantiations
  const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
  const auto a_squared =
      std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
  const auto& rho = get(cache->get_var(*this, internal_tags::rho<DataType>{}));
  const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));

  // matrix_D Calculation
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      matrix_D->get(i, j) = 1. / cube(rho) / r;
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      // Kronecker delta
      if (i == j) {
        matrix_D->get(i, j) *=
            (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, j));
      } else {
        matrix_D->get(i, j) *= -gsl::at(spin_a, i) * gsl::at(spin_a, j);
      }
    }
  }

  std::cout << "this is matrix_D:"
            << "\n"
            << *matrix_D << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_C,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::matrix_C<DataType, Frame> /*meta*/) const {
  // Instantiations
  const auto& matrix_F =
      cache->get_var(*this, internal_tags::matrix_F<DataType, Frame>{});
  const auto& matrix_D =
      cache->get_var(*this, internal_tags::matrix_D<DataType, Frame>{});

  // matrix_C Calculation
  for (size_t i = 0; i < 3; ++i) {
    for (size_t m = 0; m < 3; ++m) {
      matrix_C->get(i, m) = matrix_D.get(i, m) - 3. * matrix_F.get(i, m);
    }
  }

  std::cout << "this is matrix_C:"
            << "\n"
            << *matrix_C << "\n";
}

template <typename DataType, typename Frame>
void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
    const gsl::not_null<tnsr::ijK<DataType, 3, Frame>*> deriv_jacobian,
    const gsl::not_null<CachedBuffer*> cache,
    internal_tags::deriv_jacobian<DataType, Frame> /*meta*/) const {
  // Instantiations
  const auto& matrix_C =
      cache->get_var(*this, internal_tags::matrix_C<DataType, Frame>{});
  const auto& matrix_F =
      cache->get_var(*this, internal_tags::matrix_F<DataType, Frame>{});
  const auto& x_sph_minus_center = cache->get_var(
      *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
  const auto& r_squared =
      get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));

  // deriv_Jacobian Calculation
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        deriv_jacobian->get(i, j, k) =
            matrix_F.get(i, j) * x_sph_minus_center.get(k) +
            matrix_F.get(i, k) * x_sph_minus_center.get(j);

        for (size_t m = 0; m < 3; ++m) {
          // Kronecker delta
          if (j == k) {
            deriv_jacobian->get(i, j, k) +=
                matrix_F.get(i, m) * x_sph_minus_center.get(m);
          }
          deriv_jacobian->get(i, j, k) +=
              matrix_C.get(i, m) * x_sph_minus_center.get(k) *
              x_sph_minus_center.get(m) * x_sph_minus_center.get(j) / r_squared;
        }
      }
    }
  }

  std::cout << "this is deriv_jacobian:"
            << "\n"
            << *deriv_jacobian << "\n";
}

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_Q,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_Q<DataType, Frame> /*meta*/) const {
//   // Instantiations
//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
//   const auto& rho = get(cache->get_var(*this,
//   internal_tags::rho<DataType>{})); const auto& r = get(cache->get_var(*this,
//   internal_tags::r<DataType>{}));

//   // matrix_Q Calculation
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       matrix_Q->get(i, j) = 1. / ((rho + r) * rho);
//     }
//   }
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       // Kronecker delta
//       if (i == j) {
//         matrix_Q->get(i, j) *= gsl::at(spin_a, i) * gsl::at(spin_a, j);
//         matrix_Q->get(i, j) += r / rho;
//       } else {
//         matrix_Q->get(i, j) *= gsl::at(spin_a, i) * gsl::at(spin_a, j);
//       }
//     }
//   }
//   std::cout << "this is matrix_Q:"
//             << "\n"
//             << *matrix_Q << "\n";
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G1,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_G1<DataType, Frame> /*meta*/) const {
//   // Instantiations
//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
//   const auto a_squared =
//       std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
//   const auto& rho = get(cache->get_var(*this,
//   internal_tags::rho<DataType>{})); const auto& r = get(cache->get_var(*this,
//   internal_tags::r<DataType>{}));

//   // matrix_G1 Calculation
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t m = 0; m < 3; ++m) {
//       matrix_G1->get(i, m) = 1. / square(rho) / r;
//     }
//   }
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t m = 0; m < 3; ++m) {
//       // Kronecker delta
//       if (i == m) {
//         matrix_G1->get(i, m) *=
//             (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, m));
//       } else {
//         matrix_G1->get(i, m) *= -gsl::at(spin_a, i) * gsl::at(spin_a, m);
//       }
//     }
//   }
//   std::cout << "this is matrix_G1:"
//             << "\n"
//             << *matrix_G1 << "\n";
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<Scalar<DataType>*> a_dot_x,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::a_dot_x<DataType> /*meta*/) const {
//   // Instantiations
//   const auto& x_sph_minus_center = cache->get_var(
//       *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();

//   // a_dot_x Calculation
//   get(*a_dot_x) = spin_a[0] * get<0>(x_sph_minus_center) +
//                   spin_a[1] * get<1>(x_sph_minus_center) +
//                   spin_a[2] * get<2>(x_sph_minus_center);

//   // std::cout << "this is spin_a " << spin_a << "\n";
//   std::cout << "this is a_dot_x " << *a_dot_x << "\n";
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<Scalar<DataType>*> s_number,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::s_number<DataType> /*meta*/) const {
//   // Instantiations
//   const auto& r_squared =
//       get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
//   const auto& a_dot_x =
//       get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));

//   // s_number Calculation
//   get(*s_number) = r_squared + square(a_dot_x) / r_squared;

//   std::cout << "this is s_number:"
//             << "\n"
//             << *s_number << "\n";
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_G2,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_G2<DataType, Frame> /*meta*/) const {
//   // Instantiations
//   const auto& matrix_Q =
//       cache->get_var(*this, internal_tags::matrix_Q<DataType, Frame>{});
//   const auto& rho = get(cache->get_var(*this,
//   internal_tags::rho<DataType>{})); const auto& r = get(cache->get_var(*this,
//   internal_tags::r<DataType>{})); const auto& s_number =
//       get(cache->get_var(*this, internal_tags::s_number<DataType>{}));

//   // matrix_G2 Calculation
//   for (size_t n = 0; n < 3; ++n) {
//     for (size_t j = 0; j < 3; ++j) {
//       matrix_G2->get(n, j) = square(rho) / (s_number * r);
//     }
//   }
//   for (size_t n = 0; n < 3; ++n) {
//     for (size_t j = 0; j < 3; ++j) {
//       matrix_G2->get(n, j) *= matrix_Q.get(n, j);
//     }
//   }

//   std::cout << "this is matrix_G2:"
//             << "\n"
//             << *matrix_G2 << "\n";
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::I<DataType, 3, Frame>*> G1_dot_x,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::G1_dot_x<DataType, Frame> /*meta*/) const {
//   // Instantiations
//   const auto& x_sph_minus_center = cache->get_var(
//       *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
//   const auto& matrix_G1 =
//       cache->get_var(*this, internal_tags::matrix_G1<DataType, Frame>{});

//   // G1_dot_x Calculation
//   for (size_t i = 0; i < 3; ++i) {
//     G1_dot_x->get(i) = matrix_G1.get(i, 0) * x_sph_minus_center.get(0);
//     for (size_t m = 1; m < 3; ++m) {
//       G1_dot_x->get(i) += matrix_G1.get(i, m) * x_sph_minus_center.get(m);
//     }
//   }

//   std::cout << "this is G1_dot_x:"
//             << "\n"
//             << *G1_dot_x << "\n";
// }

// OTHER (INCORRECT) VERSION OF G1_DOT_X
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::I<DataType, 3, Frame>*> G1_dot_x,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::G1_dot_x<DataType, Frame> /*meta*/) const {
//   const auto& x_sph_minus_center = cache->get_var(
//       *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
//   const auto& matrix_G1 =
//       cache->get_var(*this, internal_tags::matrix_G1<DataType, Frame>{});

//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t m = 0; m < 3; ++m) {
//       G1_dot_x->get(i) += matrix_G1.get(i, m) * x_sph_minus_center.get(m);
//     }
//   }

//   std::cout << "this is G1_dot_x:"
//             << "\n"
//             << *G1_dot_x << "\n";
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::i<DataType, 3, Frame>*> G2_dot_x,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::G2_dot_x<DataType, Frame> /*meta*/) const {
//   // Instantiations
//   const auto& x_sph_minus_center = cache->get_var(
//       *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
//   const auto& matrix_G2 =
//       cache->get_var(*this, internal_tags::matrix_G2<DataType, Frame>{});

//   // G2_dot_x Calculation
//   for (size_t j = 0; j < 3; ++j) {
//     G2_dot_x->get(j) = matrix_G2.get(0, j) * x_sph_minus_center.get(0);
//     for (size_t n = 1; n < 3; ++n) {
//       G2_dot_x->get(j) += matrix_G2.get(n, j) * x_sph_minus_center.get(n);
//     }
//   }
//   std::cout << "this is G2_dot_x:"
//             << "\n"
//             << *G2_dot_x << "\n";
// }

// OTHER (INCORRECT) VERSION OF G2_dot_x
// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::i<DataType, 3, Frame>*> G2_dot_x,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::G2_dot_x<DataType, Frame> /*meta*/) const {
//   const auto& x_sph_minus_center =
//       cache->get_var(*this, internal_tags::x_sph_minus_center<DataType,
//       Frame>{});
//   const auto& matrix_G2 =
//       cache->get_var(*this, internal_tags::matrix_G2<DataType, Frame>{});

//   for (size_t j = 0; j < 3; ++j) {
//     for (size_t n = 0; n < 3; ++n) {
//       G2_dot_x->get(j) += matrix_G2.get(n, j) * x_sph_minus_center.get(n);
//     }
//   }
//   std::cout << "this is G2_dot_x:"
//             << "\n"
//             << *G2_dot_x << "\n";
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> inv_jacobian,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::inv_jacobian<DataType, Frame> /*meta*/) const {
//   // Instantiations
//   const auto& matrix_Q =
//       cache->get_var(*this, internal_tags::matrix_Q<DataType, Frame>{});
//   const auto& G1_dot_x =
//       cache->get_var(*this, internal_tags::G1_dot_x<DataType, Frame>{});
//   const auto& G2_dot_x =
//       cache->get_var(*this, internal_tags::G2_dot_x<DataType, Frame>{});

//   // inv_jacobian Calculation
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       inv_jacobian->get(i, j) =
//           matrix_Q.get(i, j) + G1_dot_x.get(i) * G2_dot_x.get(j);
//     }
//   }
//   std::cout << "this is inv_jacobian:"
//             << "\n"
//             << *inv_jacobian << "\n";
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_E1,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_E1<DataType, Frame> /*meta*/) const {
//   // Instantiations
//   const auto& rho = get(cache->get_var(*this,
//   internal_tags::rho<DataType>{})); const auto& r_squared =
//       get(cache->get_var(*this, internal_tags::r_squared<DataType>{}));
//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
//   const auto a_squared =
//       std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);

//   // matrix_E1 Calculation
//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t m = 0; m < 3; ++m) {
//       matrix_E1->get(i, m) =
//           -1. / square(rho) * (1. / r_squared + 2 / square(rho));
//       // Kronecker delta
//       if (i == m) {
//         matrix_E1->get(i, m) *=
//             (a_squared - gsl::at(spin_a, i) * gsl::at(spin_a, m));
//       } else {
//         matrix_E1->get(i, m) *= -gsl::at(spin_a, i) * gsl::at(spin_a, m);
//       }
//     }
//   }

//   std::cout << "this is matrix_E1:"
//             << "\n"
//             << *matrix_E1 << "\n";
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::Ij<DataType, 3, Frame>*> matrix_E2,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::matrix_E2<DataType, Frame> /*meta*/) const {
//   // Instantiations
//   const auto& rho = get(cache->get_var(*this,
//   internal_tags::rho<DataType>{})); const auto spin_a =
//   solution_.dimensionless_spin() * solution_.mass(); const auto a_squared =
//       std::inner_product(spin_a.begin(), spin_a.end(), spin_a.begin(), 0.);
//   const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
//   const auto& a_dot_x =
//       get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
//   const auto& s_number =
//       get(cache->get_var(*this, internal_tags::s_number<DataType>{}));
//   const auto& matrix_G2 =
//       cache->get_var(*this, internal_tags::matrix_G2<DataType, Frame>{});
//   const auto& matrix_P =
//       cache->get_var(*this, internal_tags::matrix_P<DataType, Frame>{});

//   // matrix_E2 Calculation
//   for (size_t n = 0; n < 3; ++n) {
//     for (size_t j = 0; j < 3; ++j) {
//       matrix_E2->get(n, j) = (1. / s_number) * matrix_P.get(n, j);
//     }
//   }
//   for (size_t n = 0; n < 3; ++n) {
//     for (size_t j = 0; j < 3; ++j) {
//       matrix_E2->get(n, j) +=
//           ((-a_squared / (square(rho) * r)) -
//            (2. / s_number) * (r - (square(a_dot_x) / cube(r)))) *
//           matrix_G2.get(n, j);
//     }
//   }

//   std::cout << "this is matrix_E2:"
//             << "\n"
//             << *matrix_E2 << "\n";
// }

// template <typename DataType, typename Frame>
// void SphKerrSchild::IntermediateComputer<DataType, Frame>::operator()(
//     const gsl::not_null<tnsr::ijK<DataType, 3, Frame>*> deriv_inv_jacobian,
//     const gsl::not_null<CachedBuffer*> cache,
//     internal_tags::deriv_inv_jacobian<DataType, Frame> /*meta*/) const {
//   // Instantiations
//   const auto spin_a = solution_.dimensionless_spin() * solution_.mass();
//   const auto& matrix_D =
//       cache->get_var(*this, internal_tags::matrix_D<DataType, Frame>{});
//   const auto& matrix_G1 =
//       cache->get_var(*this, internal_tags::matrix_G1<DataType, Frame>{});
//   const auto& matrix_G2 =
//       cache->get_var(*this, internal_tags::matrix_G2<DataType, Frame>{});
//   const auto& matrix_E1 =
//       cache->get_var(*this, internal_tags::matrix_E1<DataType, Frame>{});
//   const auto& matrix_E2 =
//       cache->get_var(*this, internal_tags::matrix_E2<DataType, Frame>{});
//   const auto& x_sph_minus_center = cache->get_var(
//       *this, internal_tags::x_sph_minus_center<DataType, Frame>{});
//   const auto& a_dot_x =
//       get(cache->get_var(*this, internal_tags::a_dot_x<DataType>{}));
//   const auto& r = get(cache->get_var(*this, internal_tags::r<DataType>{}));
//   const auto& G1_dot_x =
//       cache->get_var(*this, internal_tags::G1_dot_x<DataType, Frame>{});
//   const auto& G2_dot_x =
//       cache->get_var(*this, internal_tags::G2_dot_x<DataType, Frame>{});
//   const auto& s_number =
//       get(cache->get_var(*this, internal_tags::s_number<DataType>{}));

//   // deriv_inv_jacobian Calculation
//   for (size_t k = 0; k < 3; ++k) {
//     for (size_t i = 0; i < 3; ++i) {
//       for (size_t j = 0; j < 3; ++j) {
//         deriv_inv_jacobian->get(i, j, k) =
//             matrix_D.get(i, j) * x_sph_minus_center.get(k) +
//             matrix_G1.get(i, k) * G2_dot_x.get(j) +
//             matrix_G2.get(k, j) * G1_dot_x.get(i) -
//             2. * a_dot_x * gsl::at(spin_a, k) / s_number / square(r) *
//                 G1_dot_x.get(i) * G2_dot_x.get(j);

//         for (size_t m = 0; m < 3; ++m) {
//           deriv_inv_jacobian->get(i, j, k) +=
//               matrix_E1.get(i, m) * x_sph_minus_center.get(m) *
//                   G2_dot_x.get(j) * x_sph_minus_center.get(k) / r +
//               G1_dot_x.get(i) * x_sph_minus_center.get(k) *
//                   x_sph_minus_center.get(m) * matrix_E2.get(m, j) / r;
//         }
//       }
//     }
//   }

//   std::cout << "this is deriv_inv_jacobian:"
//             << "\n"
//             << *deriv_inv_jacobian << "\n";
// }

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template class SphKerrSchild::IntermediateVars<DTYPE(data), FRAME(data)>; \
  template class SphKerrSchild::IntermediateComputer<DTYPE(data), FRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, double),
                        (::Frame::Inertial, ::Frame::Grid))
#undef INSTANTIATE
#undef DTYPE
#undef FRAME
}  // namespace gr::Solutions
