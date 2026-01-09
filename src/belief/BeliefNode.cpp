#include "dbea/BeliefNode.h"
#include <cmath>

namespace dbea {

double BeliefNode::match_score(const PatternSignature& input) const {
    // Simple cosine-like similarity (v0)
    double score = 0.0;
    size_t n = std::min(input.features.size(), prototype.features.size());

    for (size_t i = 0; i < n; ++i) {
        score += 1.0 - std::abs(input.features[i] - prototype.features[i]);
    }

    return score / (n + 1e-6);
}

void BeliefNode::reinforce(double amount) {
    confidence += amount;
    if (confidence > 1.0) confidence = 1.0;
}

void BeliefNode::decay(double amount) {
    confidence -= amount;
    if (confidence < 0.0) confidence = 0.0;
}

}
