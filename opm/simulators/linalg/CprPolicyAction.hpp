/*
  Copyright 2024 OPM Contributors.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_CPR_POLICY_ACTION_HPP
#define OPM_CPR_POLICY_ACTION_HPP

namespace Opm {

/// Pressure-decoupling weight computation method.
enum class CprWeightType {
    QuasiIMPES,         ///< O(n) diagonal weight extraction
    TrueIMPES,          ///< O(n × numEq³) local-solve weights
    TrueIMPESAnalytic   ///< Analytic TrueIMPES variant
};

/// Smoother type for CPR fine-level and coarse AMG level.
/// Index 0 serialises as "paroverilu0" on the fine level and "ilu0" on the
/// coarse AMG level (see NeuralCprPolicy::fineSmootherStr/coarseSmootherStr) —
/// both names route to the same underlying ILU(0) smoother in the ISTL
/// preconditioner factory.
enum class CprSmoother {
    ILU0,          ///< ILU(0), parallel-overlap variant on the fine level
    DILU,          ///< Decoupled ILU
    Jacobi,        ///< Point Jacobi
    SSOR           ///< Symmetric successive over-relaxation
};

/// A discrete CPR solver configuration produced by NeuralCprPolicy::predict().
///
/// Covers five independent axes:
///   - Pressure-decoupling strategy   (weight_type)
///   - Whether to use CPRW well-weights (use_cprw)
///   - Fine-level smoother type        (fine_smoother)
///   - Coarse AMG smoother type        (coarse_smoother)
///   - Coarse-solver tolerance         (coarse_tol)
struct CprPolicyAction {
    CprWeightType weight_type     = CprWeightType::TrueIMPES;
    bool          use_cprw        = true;
    CprSmoother   fine_smoother   = CprSmoother::DILU;
    CprSmoother   coarse_smoother = CprSmoother::ILU0;
    double        coarse_tol      = 0.01;

    bool operator==(const CprPolicyAction& o) const noexcept
    {
        return weight_type     == o.weight_type
            && use_cprw        == o.use_cprw
            && fine_smoother   == o.fine_smoother
            && coarse_smoother == o.coarse_smoother
            && coarse_tol      == o.coarse_tol;
    }
    bool operator!=(const CprPolicyAction& o) const noexcept { return !(*this == o); }
};

} // namespace Opm

#endif // OPM_CPR_POLICY_ACTION_HPP
