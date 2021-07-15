// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Solutions.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace gr {
namespace Solutions {

class SphKerrSchild : public AnalyticSolution<3_st>,
                      public MarkAsAnalyticSolution {
 public:
  struct Mass {
    using type = double;
    static constexpr Options::String help = {"Mass of the black hole"};
    static type lower_bound() noexcept { return 0.; }
  };
  struct Spin {
    using type = std::array<double, volume_dim>;
    static constexpr Options::String help = {
        "The [x,y,z] dimensionless spin of the black hole"};
  };
  struct Center {
    using type = std::array<double, volume_dim>;
    static constexpr Options::String help = {
        "The [x,y,z] center of the black hole"};
  };
  using options = tmpl::list<Mass, Spin, Center>;
  static constexpr Options::String help{
      "Black hole in Spherical Kerr-Schild coordinates"};

  SphKerrSchild(double mass, Spin::type dimensionless_spin, Center::type center,
                const Options::Context& context = {});

  explicit SphKerrSchild(CkMigrateMessage* /*unused*/) noexcept {}

  SphKerrSchild() = default;
  SphKerrSchild(const SphKerrSchild& /*rhs*/) = default;
  SphKerrSchild& operator=(const SphKerrSchild& /*rhs*/) = default;
  SphKerrSchild(SphKerrSchild&& /*rhs*/) noexcept = default;
  SphKerrSchild& operator=(SphKerrSchild&& /*rhs*/) noexcept = default;
  ~SphKerrSchild() = default;

  template <typename DataType, typename Frame, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame>& x, double /*t*/,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(
        tmpl2::flat_all_v<
            tmpl::list_contains_v<tags<DataType, Frame>, Tags>...>,
        "At least one of the requested tags is not supported. The requested "
        "tags are listed as template parameters of the `variables` function.");
    IntermediateVars<DataType, Frame> intermediate(*this, x);
    return {intermediate.get_var(Tags{})...};
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  SPECTRE_ALWAYS_INLINE double mass() const noexcept { return mass_; }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>& center()
      const noexcept {
    return center_;
  }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>&
  dimensionless_spin() const noexcept {
    return dimensionless_spin_;
  }

  struct internal_tags {
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using x_sph_minus_center = ::Tags::TempI<0, 3, Frame, DataType>;
    template <typename DataType>
    using r_squared = ::Tags::TempScalar<1, DataType>;
    template <typename DataType>
    using r = ::Tags::TempScalar<2, DataType>;
    template <typename DataType>
    using rho = ::Tags::TempScalar<3, DataType>;
    template <typename DataType>
    using a_dot_x = ::Tags::TempScalar<4, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_F = ::Tags::Tempij<5, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_P = ::Tags::Tempij<6, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using jacobian = ::Tags::Tempij<7, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_D = ::Tags::Tempij<8, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_C = ::Tags::Tempij<9, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_jacobian = ::Tags::Tempij<10, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_Q = ::Tags::Tempij<11, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_G1 = ::Tags::Tempij<12, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_G2 = ::Tags::Tempij<13, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using inv_jacobian = ::Tags::Tempij<14, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_E1 = ::Tags::Tempij<15, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using matrix_E2 = ::Tags::Tempij<16, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_inv_jacobian = ::Tags::Tempij<17, 3, Frame, DataType>;
    template <typename DataType>
    using x_kerr_schild = ::Tags::TempScalar<18, DataType>;
    template <typename DataType>
    using den = ::Tags::TempScalar<19, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using a_cross_x = ::Tags::TempI<20, 3, Frame, DataType>;
    template <typename DataType>
    using a_dot_x_squared = ::Tags::TempScalar<21, DataType>;
    template <typename DataType>
    using H_denom = ::Tags::TempScalar<22, DataType>;
    template <typename DataType>
    using H = ::Tags::TempScalar<23, DataType>;
    template <typename DataType>
    using deriv_H_temp1 = ::Tags::TempScalar<24, DataType>;
    template <typename DataType>
    using deriv_H_temp2 = ::Tags::TempScalar<25, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_H = ::Tags::Tempi<26, 3, Frame, DataType>;
    template <typename DataType>
    using denom = ::Tags::TempScalar<27, DataType>;
    template <typename DataType>
    using a_dot_x_over_r = ::Tags::TempScalar<28, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using null_form = ::Tags::Tempi<29, 3, Frame, DataType>;
    template <typename DataType, typename Frame = ::Frame::Inertial>
    using deriv_null_form = ::Tags::Tempij<30, 3, Frame, DataType>;
    template <typename DataType>
    using lapse_squared = ::Tags::TempScalar<31, DataType>;
    template <typename DataType>
    using deriv_lapse_multiplier = ::Tags::TempScalar<32, DataType>;
    template <typename DataType>
    using shift_multiplier = ::Tags::TempScalar<34, DataType>;
  };

  template <typename DataType, typename Frame>
  class IntermediateComputer;

  template <typename DataType, typename Frame = ::Frame::Inertial>
  using CachedBuffer = CachedTempBuffer<
      IntermediateComputer<DataType, Frame>,
      internal_tags::x_sph_minus_center<DataType, Frame>,
      internal_tags::r_squared<DataType>, internal_tags::r<DataType>,
      internal_tags::rho<DataType>, internal_tags::a_dot_x<DataType>,
      internal_tags::matrix_F<DataType, Frame>,
      internal_tags::matrix_P<DataType, Frame>,
      internal_tags::jacobian<DataType, Frame>,
      internal_tags::matrix_D<DataType, Frame>,
      internal_tags::matrix_C<DataType, Frame>,
      internal_tags::deriv_jacobian<DataType, Frame>,
      internal_tags::matrix_Q<DataType, Frame>,
      internal_tags::matrix_G1<DataType, Frame>,
      internal_tags::matrix_G2<DataType, Frame>,
      internal_tags::inv_jacobian<DataType, Frame>,

      // missing matrix_E1, matrix_E2, deriv_inv_jacobian ???

      internal_tags::x_kerr_schild<DataType>, internal_tags::den<DataType>,
      internal_tags::a_cross_x<DataType, Frame>,
      internal_tags::a_dot_x_squared<DataType>,
      internal_tags::H_denom<DataType>, internal_tags::H<DataType>,
      internal_tags::deriv_H_temp1<DataType>,
      internal_tags::deriv_H_temp2<DataType>,
      internal_tags::deriv_H<DataType, Frame>, internal_tags::denom<DataType>,
      internal_tags::a_dot_x_over_r<DataType>,
      internal_tags::null_form<DataType, Frame>,
      internal_tags::deriv_null_form<DataType, Frame>,
      internal_tags::lapse_squared<DataType>, gr::Tags::Lapse<DataType>,
      internal_tags::deriv_lapse_multiplier<DataType>,
      internal_tags::shift_multiplier<DataType>,
      gr::Tags::Shift<3, Frame, DataType>, DerivShift<DataType, Frame>,
      gr::Tags::SpatialMetric<3, Frame, DataType>,
      DerivSpatialMetric<DataType, Frame>,
      ::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>>>;

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateComputer {
   public:
    using CachedBuffer = SphKerrSchild::CachedBuffer<DataType, Frame>;

    IntermediateComputer(const SphKerrSchild& solution,
                         const tnsr::I<DataType, 3, Frame>& x,
                         double null_vector_0) noexcept;

    void operator()(
        gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_sph_minus_center,
        gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::x_sph_minus_center<DataType, Frame> /*meta*/)
        const noexcept;

    void operator()(gsl::not_null<Scalar<DataType>*> r_squared,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r_squared<DataType> /*meta*/) const noexcept;

    void operator()(gsl::not_null<Scalar<DataType>*> r,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r<DataType> /*meta*/) const noexcept;

    void operator()(gsl::not_null<Scalar<DataType>*> rho,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::rho<DataType> /*meta*/) const noexcept;

    void operator()(gsl::not_null<Scalar<DataType>*> a_dot_x,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::a_dot_x<DataType> /*meta*/) const noexcept;

    void operator()(
        gsl::not_null<tnsr::ij<DataType, 3, Frame>*> matrix_F,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::matrix_F<DataType, Frame> /*meta*/) const noexcept;

    void operator()(
        gsl::not_null<tnsr::ij<DataType, 3, Frame>*> matrix_P,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::matrix_P<DataType, Frame> /*meta*/) const noexcept;

    void operator()(
        gsl::not_null<tnsr::ij<DataType, 3, Frame>*> jacobian,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::jacobian<DataType, Frame> /*meta*/) const noexcept;

    void operator()(
        gsl::not_null<tnsr::ij<DataType, 3, Frame>*> matrix_D,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::matrix_D<DataType, Frame> /*meta*/) const noexcept;

    void operator()(
        gsl::not_null<tnsr::ij<DataType, 3, Frame>*> matrix_C,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::matrix_C<DataType, Frame> /*meta*/) const noexcept;

    void operator()(
        gsl::not_null<tnsr::ij<DataType, 3, Frame>*> deriv_jacobian,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::deriv_jacobian<DataType, Frame> /*meta*/) const noexcept;

    void operator()(
        gsl::not_null<tnsr::ij<DataType, 3, Frame>*> matrix_Q,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::matrix_Q<DataType, Frame> /*meta*/) const noexcept;

    void operator()(
        gsl::not_null<tnsr::ij<DataType, 3, Frame>*> matrix_G1,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::matrix_G1<DataType, Frame> /*meta*/) const noexcept;

    void operator()(
        gsl::not_null<tnsr::ij<DataType, 3, Frame>*> matrix_G2,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::matrix_G2<DataType, Frame> /*meta*/) const noexcept;

    void operator()(
        gsl::not_null<tnsr::ij<DataType, 3, Frame>*> inv_jacobian,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::inv_jacobian<DataType, Frame> /*meta*/) const noexcept;

    void operator()(
        gsl::not_null<Scalar<DataType>*> x_kerr_schild,
        gsl::not_null<CachedBuffer*> cache,
        internal_tags::x_kerr_schild<DataType> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<Scalar<DataType>*> den,
    //     gsl::not_null<CachedBuffer*> cache,
    //     internal_tags::a_dot_x_squared<DataType> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<tnsr::i<DataType, 3, Frame>*> a_cross_x,
    //     gsl::not_null<CachedBuffer*> cache,
    //     internal_tags::a_dot_x_over_rsquared<DataType> /*meta*/) const
    //     noexcept;

    // void operator()(
    //     gsl::not_null<Scalar<DataType>*> a_dot_x_squared,
    //     gsl::not_null<CachedBuffer*> cache,
    //     internal_tags::a_dot_x_squared<DataType> /*meta*/) const noexcept;

    // void operator()(gsl::not_null<Scalar<DataType>*> H_denom,
    //                 gsl::not_null<CachedBuffer*> cache,
    //                 internal_tags::H_denom<DataType> /*meta*/) const
    //                 noexcept;

    // void operator()(gsl::not_null<Scalar<DataType>*> H,
    //                 gsl::not_null<CachedBuffer*> cache,
    //                 internal_tags::H<DataType> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<Scalar<DataType>*> deriv_H_temp1,
    //     gsl::not_null<CachedBuffer*> cache,
    //     internal_tags::deriv_H_temp1<DataType> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<Scalar<DataType>*> deriv_H_temp2,
    //     gsl::not_null<CachedBuffer*> cache,
    //     internal_tags::deriv_H_temp2<DataType> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<tnsr::i<DataType, 3, Frame>*> deriv_H,
    //     gsl::not_null<CachedBuffer*> cache,
    //     internal_tags::deriv_H<DataType, Frame> /*meta*/) const noexcept;

    // void operator()(gsl::not_null<Scalar<DataType>*> denom,
    //                 gsl::not_null<CachedBuffer*> cache,
    //                 internal_tags::denom<DataType> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<Scalar<DataType>*> a_dot_x_over_r,
    //     gsl::not_null<CachedBuffer*> cache,
    //     internal_tags::a_dot_x_over_r<DataType> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<tnsr::i<DataType, 3, Frame>*> null_form,
    //     gsl::not_null<CachedBuffer*> cache,
    //     internal_tags::null_form<DataType, Frame> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<tnsr::ij<DataType, 3, Frame>*> deriv_null_form,
    //     gsl::not_null<CachedBuffer*> cache,
    //     internal_tags::deriv_null_form<DataType, Frame> /*meta*/)
    //     const noexcept;

    // void operator()(
    //     gsl::not_null<Scalar<DataType>*> lapse_squared,
    //     gsl::not_null<CachedBuffer*> cache,
    //     internal_tags::lapse_squared<DataType> /*meta*/) const noexcept;

    // void operator()(gsl::not_null<Scalar<DataType>*> lapse,
    //                 gsl::not_null<CachedBuffer*> cache,
    //                 gr::Tags::Lapse<DataType> /*meta*/) const noexcept;

    // void operator()(gsl::not_null<Scalar<DataType>*> deriv_lapse_multiplier,
    //                 gsl::not_null<CachedBuffer*> cache,
    //                 internal_tags::deriv_lapse_multiplier<DataType> /*meta*/)
    //     const noexcept;

    // void operator()(
    //     gsl::not_null<Scalar<DataType>*> shift_multiplier,
    //     gsl::not_null<CachedBuffer*> cache,
    //     internal_tags::shift_multiplier<DataType> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<tnsr::I<DataType, 3, Frame>*> shift,
    //     gsl::not_null<CachedBuffer*> cache,
    //     gr::Tags::Shift<3, Frame, DataType> /*meta*/) const noexcept;

    // void operator()(gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> deriv_shift,
    //                 gsl::not_null<CachedBuffer*> cache,
    //                 DerivShift<DataType, Frame> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
    //     gsl::not_null<CachedBuffer*> cache,
    //     gr::Tags::SpatialMetric<3, Frame, DataType> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> deriv_spatial_metric,
    //     gsl::not_null<CachedBuffer*> cache,
    //     DerivSpatialMetric<DataType, Frame> /*meta*/) const noexcept;

    // void operator()(
    //     gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dt_spatial_metric,
    //     gsl::not_null<CachedBuffer*> cache,
    //     ::Tags::dt<gr::Tags::SpatialMetric<3, Frame, DataType>> /*meta*/)
    //     const noexcept;

   private:
    const SphKerrSchild& solution_;
    const tnsr::I<DataType, 3, Frame>& x_;
    double null_vector_0_;
  };

  template <typename DataType, typename Frame = ::Frame::Inertial>
  class IntermediateVars : public CachedBuffer<DataType, Frame> {
   public:
    using CachedBuffer = SphKerrSchild::CachedBuffer<DataType, Frame>;

    IntermediateVars(const SphKerrSchild& solution,
                     const tnsr::I<DataType, 3, Frame>& x) noexcept;

    using CachedBuffer::get_var;

    tnsr::i<DataType, 3, Frame> get_var(
        DerivLapse<DataType, Frame> /*meta*/) noexcept;

    Scalar<DataType> get_var(
        ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/) noexcept;

    tnsr::I<DataType, 3, Frame> get_var(
        ::Tags::dt<gr::Tags::Shift<3, Frame, DataType>> /*meta*/) noexcept;

    Scalar<DataType> get_var(
        gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/) noexcept;

    tnsr::II<DataType, 3, Frame> get_var(
        gr::Tags::InverseSpatialMetric<3, Frame, DataType> /*meta*/) noexcept;

    tnsr::ii<DataType, 3, Frame> get_var(
        gr::Tags::ExtrinsicCurvature<3, Frame, DataType> /*meta*/) noexcept;

   private:
    // Here null_vector_0 is simply -1, but if you have a boosted solution,
    // then null_vector_0 can be something different, so we leave it coded
    // in instead of eliminating it.
    static constexpr double null_vector_0_ = -1.0;
  };

 private:
  double mass_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, volume_dim> dimensionless_spin_ =
      make_array<volume_dim>(std::numeric_limits<double>::signaling_NaN());
  std::array<double, volume_dim> center_ =
      make_array<volume_dim>(std::numeric_limits<double>::signaling_NaN());
};

SPECTRE_ALWAYS_INLINE bool operator==(const SphKerrSchild& lhs,
                                      const SphKerrSchild& rhs) noexcept {
  return lhs.mass() == rhs.mass() and
         lhs.dimensionless_spin() == rhs.dimensionless_spin() and
         lhs.center() == rhs.center();
}

SPECTRE_ALWAYS_INLINE bool operator!=(const SphKerrSchild& lhs,
                                      const SphKerrSchild& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace Solutions
}  // namespace gr
