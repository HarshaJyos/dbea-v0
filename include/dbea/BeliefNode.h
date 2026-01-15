#pragma once
#include <string>
#include <unordered_map>
#include <vector> // for PatternSignature
#include "dbea/PatternSignature.h"
namespace dbea
{
    struct BeliefNode
    {
        std::string id;
        PatternSignature prototype;
        double confidence;
        double activation;
        int evidence_count = 1;
        double last_predicted_reward = 0.0;
        double prediction_error = 0.0;
        std::unordered_map<int, double> action_values;
        double fitness = 0.0;  // NEW: Cumulative causal contribution
        double mutation_rate = 0.1;  // NEW: Heritable, adaptive
        double local_lr;  // NEW: Belief-specific learning rate
        std::vector<double> emotional_affinity;  // NEW: [5] for emotion biases
        BeliefNode(const std::string &id_,
                   const PatternSignature &proto);
        double match_score(const PatternSignature &input) const;
        void reinforce(double amount);
        void decay(double amount);
        double predict_action_value(int action_id) const;
        void learn_action_value(int action_id, double reward, double learning_rate, double gamma = 0.95);
    };
} // namespace dbea
