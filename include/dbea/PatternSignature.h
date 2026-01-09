#pragma once
#include <vector>

struct PatternSignature {
    std::vector<double> features;

    PatternSignature() = default;
    explicit PatternSignature(const std::vector<double>& feats);
};
