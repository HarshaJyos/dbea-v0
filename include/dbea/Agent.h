#pragma once
#include "dbea/BeliefGraph.h"
#include "dbea/EmotionState.h"
#include "dbea/Environment.h"
#include "dbea/Action.h"
#include "dbea/PatternSignature.h"
#include "dbea/Config.h"
#include <vector>
#include <utility>

namespace dbea {

class Agent {
public:
    Agent(const Config& cfg);
    void perceive(const PatternSignature& input);
    Action decide();
    void receive_reward(double reward_valence, double reward_surprise);
    void learn();

    // NEW: Public method to prune beliefs safely
    void prune_beliefs(double threshold = 0.40) {
        belief_graph.prune(threshold);
    }

    std::pair<double, double> get_proto_action_values() const {
        if (belief_graph.nodes.empty()) return {0.0, 0.0};
        const auto& proto = belief_graph.nodes.front();
        return {proto->predict_action_value(0),
                proto->predict_action_value(1)};
    }

    size_t get_belief_count() const {
        return belief_graph.nodes.size();
    }
    
    std::vector<std::pair<double, double>> get_all_belief_action_values() const {
        std::vector<std::pair<double, double>> values;
        for (const auto& belief : belief_graph.nodes) {
            values.emplace_back(
                belief->predict_action_value(0),
                belief->predict_action_value(1)
            );
        }
        return values;
    }

private:
    Config config;
    BeliefGraph belief_graph;
    EmotionState emotion;
    std::vector<Action> available_actions;

    double last_reward = 0.0;
    Action last_action;  // Declared here
    PatternSignature last_perception;
    double last_predicted_reward = 0.0;  // NEW
};

} // namespace dbea
