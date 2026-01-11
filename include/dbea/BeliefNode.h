#pragma once
#include <string>
#include <unordered_map>
#include <vector>       // for PatternSignature
#include "dbea/PatternSignature.h"

namespace dbea {

struct BeliefNode {
    std::string id;
    PatternSignature prototype;
    double confidence;
    double activation;

    std::unordered_map<int, double> action_values;

    BeliefNode(const std::string& id_,
               const PatternSignature& proto);

    double match_score(const PatternSignature& input) const;
    void reinforce(double amount);
    void decay(double amount);

    double predict_action_value(int action_id) const;
    void learn_action_value(int action_id, double reward, double learning_rate);
};

} // namespace dbea