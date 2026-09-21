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

#ifndef OPM_NEURAL_CPR_POLICY_HPP
#define OPM_NEURAL_CPR_POLICY_HPP

#include <opm/simulators/linalg/CprPolicyAction.hpp>
#include <opm/simulators/linalg/CprPolicyFeatures.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>
#include <opm/ml/ml_model.hpp>

#include <bitset>
#include <string>
#include <vector>

namespace Opm {

/// Neural CPR policy backed by the OPM Kerasify ML framework.
///
/// Two operating modes:
///   MLP inference  — when a valid Kerasify model file is loaded
///                    (isLoaded() == true).  Runs a 12→H→288 MLP via
///                    Opm::ML::NNModel<float>.
///   Rule-based     — deterministic heuristic based on Newton iteration
///                    index and residual reduction ratio; no file I/O.
///
/// The model file is produced by the companion Python export script using
/// opm-common's opm.ml.ml_tools.kerasify.export_model().
///
/// MLP architecture: Dense(12→H, relu) → Dense(H→288, linear)
///
/// Output: 288 logits over the joint 3×2×4×4×3 action space.
/// Decode: argmax index i →
///   w=i/96, cprw=(i/48)%2, fine=(i/12)%4, coarse=(i/3)%4, tol=i%3
class NeuralCprPolicy {
public:
    static constexpr int kNumLabels = 288;

    /// Construct.  If model_path is empty or the file cannot be loaded,
    /// falls back to rule-based prediction (isLoaded() == false).
    explicit NeuralCprPolicy(const std::string& model_path = "");

    ~NeuralCprPolicy() = default;

    NeuralCprPolicy(const NeuralCprPolicy&)            = delete;
    NeuralCprPolicy& operator=(const NeuralCprPolicy&) = delete;

    bool isLoaded() const { return valid_; }

    /// Forbid a label index (0–287) from ever being selected by predict().
    /// When the model's top logit falls on a forbidden label, the next highest
    /// non-forbidden label is returned instead.  Safe to call multiple times.
    void forbidLabel(int label)
    {
        if (label >= 0 && label < kNumLabels)
            forbidden_mask_.set(static_cast<std::size_t>(label));
    }

    /// Predict the best CPR configuration for the given features.
    /// If \p out_confidence is non-null it receives the softmax probability of
    /// the top label (0–1).  Rule-based mode always writes 1.0.
    /// Forbidden labels (see forbidLabel()) are never returned.
    CprPolicyAction predict(const CprPolicyFeatures& feat,
                            float* out_confidence = nullptr) const;

    /// Encode a CprPolicyAction as its label index 0–287 (inverse of decodeLabel).
    /// Shared by ISTLSolver.hpp for parallel-broadcast and log-string purposes,
    /// so the encoding lives in exactly one place.
    static int encodeLabel(const CprPolicyAction& act);

    /// Decode a label index 0–287 into a CprPolicyAction.
    static CprPolicyAction decodeLabel(int label);

    /// Build a complete PropertyTree mirroring setupCPRW() in
    /// setupPropertyTree.cpp, accepted by FlexibleSolver without modification.
    static PropertyTree configToPropertyTree(const CprPolicyAction& act,
                                             double tol     = 0.005,
                                             int    maxiter = 20);

    /// Patch only the 1–3 fields that changed; avoids a full 30-node ptree rebuild.
    /// Precondition: act.use_cprw == prev.use_cprw.
    static void updatePropertyTreeInPlace(const CprPolicyAction& act,
                                          const CprPolicyAction& prev,
                                          PropertyTree& prm);

    static std::string weightTypeStr(CprWeightType w);
    static std::string fineSmootherStr(CprSmoother s);
    static std::string coarseSmootherStr(CprSmoother s);

private:
    mutable Opm::ML::NNModel<float> model_;
    bool     valid_            = false;
    int      model_input_size_ = CprPolicyFeatures::kNumFeatures;
    std::bitset<kNumLabels> forbidden_mask_;  ///< bitmask of labels never to select

    /// Read the input feature count from the first Dense layer of a binary model file.
    /// Returns CprPolicyFeatures::kNumFeatures on any parse failure.
    static int readModelInputSize(const std::string& path);

    static CprPolicyAction ruleBasedPredict(const CprPolicyFeatures& feat);
};

} // namespace Opm

#endif // OPM_NEURAL_CPR_POLICY_HPP
