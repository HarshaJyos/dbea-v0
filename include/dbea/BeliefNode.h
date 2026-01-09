#pragma once
#include <string>
#include <unordered_map>
#include "dbea/PatternSignature.h"

namespace dbea {

class BeliefNode {
public:
    std::string id;
    PatternSignature prototype;

    double confidence = 0.5;
    double activation = 0.0;

    // NEW: expected utility per action
    std::unordered_map<int, double> action_values;

    BeliefNode(const std::string& id,
               const PatternSignature& proto);

    double match_score(const PatternSignature& input) const;

    void reinforce(double amount);
    void decay(double amount);

    // NEW
    double predict_action_value(int action_id) const;
};

}
