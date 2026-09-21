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

#include <opm/simulators/linalg/NeuralCprPolicy.hpp>

#include <opm/common/OpmLog/OpmLog.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

namespace Opm {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

// Binary model layout (export_binary_fallback in train_cpr_policy.py):
//   [num_layers : uint32]
//   [layer_type : uint32]  (3 = Dense)
//   [weights_rows : uint32]  ← input feature count
//   ...
// We read up to the first Dense layer and return weights_rows.
int NeuralCprPolicy::readModelInputSize(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        return CprPolicyFeatures::kNumFeatures;

    auto readU32 = [&](unsigned int& v) {
        f.read(reinterpret_cast<char*>(&v), 4);
        return !f.fail();
    };

    unsigned int num_layers = 0;
    if (!readU32(num_layers) || num_layers == 0)
        return CprPolicyFeatures::kNumFeatures;

    // Scan layers until we hit the first Dense (type=3).
    // Activation layers (type=4) contain only 1 uint32; skip them.
    for (unsigned int i = 0; i < num_layers; ++i) {
        unsigned int layer_type = 0;
        if (!readU32(layer_type))
            break;
        if (layer_type == 3) {  // kDense
            unsigned int weights_rows = 0;
            if (readU32(weights_rows) && weights_rows > 0)
                return static_cast<int>(weights_rows);
            break;
        }
        if (layer_type == 4) {  // kActivation — skip its 1 uint32
            unsigned int dummy = 0;
            readU32(dummy);
        }
        // Scaling/UnScaling layers are not used by our models; stop scanning.
        else break;
    }
    return CprPolicyFeatures::kNumFeatures;
}

NeuralCprPolicy::NeuralCprPolicy(const std::string& model_path)
{
    if (model_path.empty())
        return;

    // NNModel::loadModel uses OPM_ERROR_IF which throws on file-open failure.
    try {
        valid_ = model_.loadModel(model_path);
    } catch (const std::exception& e) {
        OpmLog::warning("NeuralCprPolicy: failed to load model from '"
                        + model_path + "': " + std::string(e.what())
                        + " — using rule-based fallback");
        valid_ = false;
        return;
    }

    if (valid_) {
        model_input_size_ = readModelInputSize(model_path);
        const int nf = CprPolicyFeatures::kNumFeatures;
        OpmLog::info("NeuralCprPolicy: loaded model from " + model_path
                     + " (input=" + std::to_string(model_input_size_)
                     + "/" + std::to_string(nf) + " features"
                     + (model_input_size_ < nf ? ", older model — trailing features zeroed" : "")
                     + ")");
        if (model_input_size_ > nf) {
            OpmLog::warning("NeuralCprPolicy: model expects " + std::to_string(model_input_size_)
                            + " features but binary only provides " + std::to_string(nf)
                            + " — using rule-based fallback");
            valid_ = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

CprPolicyAction NeuralCprPolicy::predict(const CprPolicyFeatures& feat,
                                          float* out_confidence) const
{
    if (!valid_) {
        if (out_confidence) *out_confidence = 1.0f;
        return ruleBasedPredict(feat);
    }

    auto arr = feat.toArray();
    Opm::ML::Tensor<float> in(model_input_size_);
    for (int i = 0; i < model_input_size_; ++i)
        in(i) = arr[i];

    Opm::ML::Tensor<float> out;
    if (!model_.apply(in, out) || static_cast<int>(out.data_.size()) < kNumLabels) {
        if (out_confidence) *out_confidence = 1.0f;
        return ruleBasedPredict(feat);
    }

    // Apply forbidden mask: set disallowed logits to -inf so they are never
    // selected by argmax or included in the softmax normalisation sum.
    for (int i = 0; i < kNumLabels; ++i) {
        if (forbidden_mask_.test(static_cast<std::size_t>(i)))
            out.data_[i] = -std::numeric_limits<float>::infinity();
    }

    const auto* d = out.data_.data();
    const int best = static_cast<int>(std::max_element(d, d + kNumLabels) - d);

    if (out_confidence) {
        // Softmax over allowed logits only, numerically stable.
        float max_l = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < kNumLabels; ++i)
            if (std::isfinite(d[i])) max_l = std::max(max_l, d[i]);
        float sum = 0.f;
        for (int i = 0; i < kNumLabels; ++i)
            if (std::isfinite(d[i])) sum += std::exp(d[i] - max_l);
        *out_confidence = (sum > 0.f) ? std::exp(d[best] - max_l) / sum : 1.0f;
    }

    return decodeLabel(best);
}

namespace {
constexpr double kCoarseTolValues[3] = {0.01, 0.1, 0.5};

int coarseTolIndex(double tol)
{
    // Snap to the nearest of the three trained tolerance buckets.
    int best = 0;
    double bestDist = std::abs(tol - kCoarseTolValues[0]);
    for (int i = 1; i < 3; ++i) {
        const double dist = std::abs(tol - kCoarseTolValues[i]);
        if (dist < bestDist) { bestDist = dist; best = i; }
    }
    return best;
}

int smootherIndex(CprSmoother s)
{
    switch (s) {
    case CprSmoother::ILU0:   return 0;
    case CprSmoother::DILU:   return 1;
    case CprSmoother::Jacobi: return 2;
    case CprSmoother::SSOR:   return 3;
    }
    return 0;
}

CprSmoother smootherFromIndex(int i)
{
    switch (i) {
    case 0:  return CprSmoother::ILU0;
    case 1:  return CprSmoother::DILU;
    case 2:  return CprSmoother::Jacobi;
    default: return CprSmoother::SSOR;
    }
}
} // namespace

int NeuralCprPolicy::encodeLabel(const CprPolicyAction& act)
{
    const int w      = static_cast<int>(act.weight_type);
    const int cprw   = act.use_cprw ? 1 : 0;
    const int fine   = smootherIndex(act.fine_smoother);
    const int coarse = smootherIndex(act.coarse_smoother);
    const int tol    = coarseTolIndex(act.coarse_tol);
    return w * 96 + cprw * 48 + fine * 12 + coarse * 3 + tol;
}

CprPolicyAction NeuralCprPolicy::decodeLabel(int label)
{
    label = std::clamp(label, 0, kNumLabels - 1);
    const int w      = label / 96;
    int rem          = label % 96;
    const int cprw   = rem / 48;
    rem              = rem % 48;
    const int fine   = rem / 12;
    rem              = rem % 12;
    const int coarse = rem / 3;
    const int tol    = rem % 3;

    CprPolicyAction act;
    switch (w) {
    case 0:  act.weight_type = CprWeightType::QuasiIMPES;        break;
    case 1:  act.weight_type = CprWeightType::TrueIMPES;         break;
    default: act.weight_type = CprWeightType::TrueIMPESAnalytic; break;
    }
    act.use_cprw        = (cprw == 1);
    act.fine_smoother   = smootherFromIndex(fine);
    act.coarse_smoother = smootherFromIndex(coarse);
    act.coarse_tol      = kCoarseTolValues[tol];
    return act;
}

CprPolicyAction NeuralCprPolicy::ruleBasedPredict(const CprPolicyFeatures& feat)
{
    CprPolicyAction act;
    act.use_cprw        = true;
    act.coarse_smoother = CprSmoother::ILU0;

    const bool slowConv = (feat.nl_iteration > 0 && feat.nl_residual_reduce > 0.5);
    const bool manyIter = (feat.nl_iteration >= 4);

    if (slowConv || manyIter) {
        // Struggling to converge — use the most robust (expensive) config.
        act.weight_type   = CprWeightType::TrueIMPES;
        act.fine_smoother = CprSmoother::DILU;
    } else if (feat.nl_iteration == 0) {
        // First Newton step: prefer cheap weight computation and smoother setup.
        act.weight_type   = CprWeightType::QuasiIMPES;
        act.fine_smoother = CprSmoother::ILU0;
    } else {
        // Mid-convergence: balanced choice.
        act.weight_type   = CprWeightType::QuasiIMPES;
        act.fine_smoother = CprSmoother::DILU;
    }
    return act;
}

// ---------------------------------------------------------------------------
// PropertyTree helpers
// ---------------------------------------------------------------------------

PropertyTree NeuralCprPolicy::configToPropertyTree(const CprPolicyAction& act,
                                                    double tol, int maxiter)
{
    using namespace std::string_literals;
    PropertyTree prm;

    prm.put("maxiter",   maxiter);
    prm.put("tol",       tol);
    prm.put("verbosity", 0);
    prm.put("solver",    "bicgstab"s);

    prm.put("preconditioner.type",             act.use_cprw ? "cprw"s : "cpr"s);
    prm.put("preconditioner.use_well_weights", act.use_cprw ? "true"s : "false"s);
    prm.put("preconditioner.add_wells",        act.use_cprw ? "true"s : "false"s);
    prm.put("preconditioner.weight_type",      weightTypeStr(act.weight_type));
    prm.put("preconditioner.pre_smooth",       0);
    prm.put("preconditioner.post_smooth",      1);
    prm.put("preconditioner.finesmoother.type",       fineSmootherStr(act.fine_smoother));
    prm.put("preconditioner.finesmoother.relaxation", 1.0);
    prm.put("preconditioner.verbosity",        0);
    prm.put("preconditioner.coarsesolver.maxiter",   1);
    prm.put("preconditioner.coarsesolver.tol",       act.coarse_tol);
    prm.put("preconditioner.coarsesolver.solver",    "loopsolver"s);
    prm.put("preconditioner.coarsesolver.verbosity", 0);

    const std::string amg = "preconditioner.coarsesolver.preconditioner.";
    prm.put(amg + "type",                "amg"s);
    prm.put(amg + "smoother",            coarseSmootherStr(act.coarse_smoother));
    prm.put(amg + "alpha",               0.333333333333);
    prm.put(amg + "relaxation",          1.0);
    prm.put(amg + "iterations",          1);
    prm.put(amg + "coarsenTarget",       1200);
    prm.put(amg + "pre_smooth",          1);
    prm.put(amg + "post_smooth",         1);
    prm.put(amg + "beta",                0.0);
    prm.put(amg + "verbosity",           0);
    prm.put(amg + "maxlevel",            15);
    prm.put(amg + "skip_isolated",       0);
    prm.put(amg + "accumulate",          1);
    prm.put(amg + "prolongationdamping", 1.0);
    prm.put(amg + "maxdistance",         2);
    prm.put(amg + "maxconnectivity",     15);
    prm.put(amg + "maxaggsize",          6);
    prm.put(amg + "minaggsize",          4);
    return prm;
}

void NeuralCprPolicy::updatePropertyTreeInPlace(const CprPolicyAction& act,
                                                 const CprPolicyAction& prev,
                                                 PropertyTree& prm)
{
    if (act.weight_type != prev.weight_type)
        prm.put("preconditioner.weight_type", weightTypeStr(act.weight_type));
    if (act.fine_smoother != prev.fine_smoother)
        prm.put("preconditioner.finesmoother.type", fineSmootherStr(act.fine_smoother));
    if (act.coarse_smoother != prev.coarse_smoother)
        prm.put("preconditioner.coarsesolver.preconditioner.smoother",
                coarseSmootherStr(act.coarse_smoother));
    if (act.coarse_tol != prev.coarse_tol)
        prm.put("preconditioner.coarsesolver.tol", act.coarse_tol);
}

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

std::string NeuralCprPolicy::weightTypeStr(CprWeightType w)
{
    switch (w) {
    case CprWeightType::QuasiIMPES:        return "quasiimpes";
    case CprWeightType::TrueIMPES:         return "trueimpes";
    case CprWeightType::TrueIMPESAnalytic: return "trueimpesanalytic";
    }
    return "trueimpes";
}

std::string NeuralCprPolicy::fineSmootherStr(CprSmoother s)
{
    switch (s) {
    case CprSmoother::ILU0:   return "paroverilu0";
    case CprSmoother::DILU:   return "dilu";
    case CprSmoother::Jacobi: return "jac";
    case CprSmoother::SSOR:   return "ssor";
    }
    return "paroverilu0";
}

std::string NeuralCprPolicy::coarseSmootherStr(CprSmoother s)
{
    switch (s) {
    case CprSmoother::ILU0:   return "ilu0";
    case CprSmoother::DILU:   return "dilu";
    case CprSmoother::Jacobi: return "jac";
    case CprSmoother::SSOR:   return "ssor";
    }
    return "ilu0";
}

} // namespace Opm
