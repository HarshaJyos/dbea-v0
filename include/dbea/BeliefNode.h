#pragma once
#include <string>
#include <unordered_map>
#include "dbea/PatternSignature.h"

namespace dbea {

struct BeliefNode {
    std::string id;
    PatternSignature prototype;

    double confidence;
    double activation;

    // NEW: action value table
    std::unordered_map<int, double> action_values;

    BeliefNode(const std::string& id_,
               const PatternSignature& proto)
        : id(id_), prototype(proto),
          confidence(0.5), activation(0.0) {}

    double match_score(const PatternSignature& input) const;

    void reinforce(double amount);
    void decay(double amount);

    // NEW: predict value for an action
    double predict_action_value(int action_id) const;

    // NEW: update action value from reward
    void learn_action_value(int action_id, double reward, double learning_rate);
};

} // namespace dbea
