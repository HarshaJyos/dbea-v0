#include "dbea/BeliefNode.h"
#include <cmath>

namespace dbea {

// Constructor definition
BeliefNode::BeliefNode(const std::string& id_, const PatternSignature& proto)
    : id(id_), prototype(proto), confidence(0.5), activation(0.0)
{
    // Initialize empty action_values map
    action_values.clear();
}

double BeliefNode::match_score(const PatternSignature& input) const {
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

// Helper to get predicted value for an action
double BeliefNode::predict_action_value(int action_id) const {
    auto it = action_values.find(action_id);
    if (it != action_values.end()) return it->second;
    return 0.0;
}

} // namespace dbea
