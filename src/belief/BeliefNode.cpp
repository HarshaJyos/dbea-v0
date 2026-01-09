#include "dbea/BeliefNode.h"
#include <cmath>
#include <algorithm>

namespace dbea {

double BeliefNode::match_score(const PatternSignature& input) const {
    double score = 0.0;
    size_t n = std::min(input.features.size(), prototype.features.size());
    for (size_t i = 0; i < n; ++i) {
        score += 1.0 - std::abs(input.features[i] - prototype.features[i]);
    }
    return score / (n + 1e-6);
}

void BeliefNode::reinforce(double amount) {
    confidence = std::min(1.0, confidence + amount);
}

void BeliefNode::decay(double amount) {
    confidence = std::max(0.0, confidence - amount);
}

double BeliefNode::predict_action_value(int action_id) const {
    auto it = action_values.find(action_id);
    if (it != action_values.end()) {
        return it->second;
    }
    return 0.0;
}


void BeliefNode::learn_action_value(int action_id, double reward, double learning_rate) {
    double old_value = action_values[action_id];
    double new_value = old_value + learning_rate * (reward - old_value);
    action_values[action_id] = new_value;
}

} // namespace dbea
