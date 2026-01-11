#pragma once
#include "dbea/BeliefGraph.h"
#include "dbea/EmotionState.h"
#include "dbea/Environment.h"
#include "dbea/Action.h"
#include "dbea/PatternSignature.h"
#include "dbea/Config.h"
#include <vector>
#include <utility> // for std::pair

namespace dbea {

class Agent {
public:
    Agent(const Config& cfg);
    void perceive(const PatternSignature& input);
    Action decide();
    void receive_reward(double reward_valence, double reward_surprise);
    void learn();

    // Option 2: getter for proto-belief action values
    std::pair<double,double> get_proto_action_values() const {
        if (belief_graph.nodes.empty()) return {0.0, 0.0};
        const auto& proto = belief_graph.nodes.front();
        return {proto->predict_action_value(0), proto->predict_action_value(1)};
    }

private:
    Config config;
    BeliefGraph belief_graph;
    EmotionState emotion;
    std::vector<Action> available_actions;
    
    // Store the last reward received
    double last_reward = 0.0;

    // ✅ NEW: store last chosen action
    Action last_action;
};

} // namespace dbea
