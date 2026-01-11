#include "dbea/BeliefNode.h"
#include <cmath>
#include <algorithm>
#include <string>

namespace dbea {

BeliefNode::BeliefNode(const std::string& id_, const PatternSignature& proto)
    : id(id_),
      prototype(proto),
      confidence(id_ == "proto-belief" ? 0.3 : 0.5),
      activation(0.0),
      evidence_count(1),
      last_predicted_reward(0.0),     // NEW
      prediction_error(0.0)           // NEW
{
}

double BeliefNode::match_score(const PatternSignature& input) const {
    if (input.features.size() != prototype.features.size() || input.features.empty()) {
        return 0.0;
    }

    double sum_sq_diff = 0.0;
    double max_range = 0.0;

    for (size_t i = 0; i < input.features.size(); ++i) {
        double diff = input.features[i] - prototype.features[i];
        sum_sq_diff += diff * diff;
        max_range += 1.0;
    }

    double dist = std::sqrt(sum_sq_diff);
    double normalized = 1.0 - (dist / std::sqrt(max_range));

    return std::max(0.0, normalized);
}

void BeliefNode::reinforce(double amount) {
    confidence = std::min(1.0, confidence + amount);
    evidence_count++;
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

void BeliefNode::learn_action_value(int action_id,
                                   double reward,
                                   double learning_rate) {
    double old_value = action_values[action_id];
    double new_value = old_value + learning_rate * (reward - old_value);
    action_values[action_id] = new_value;
}

} // namespace dbea