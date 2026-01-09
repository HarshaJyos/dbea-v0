#pragma once
#include "dbea/BeliefGraph.h"
#include "dbea/EmotionState.h"
#include "dbea/Environment.h"
#include "dbea/Action.h"
#include "dbea/PatternSignature.h"
#include "dbea/Config.h"
#include <vector>

namespace dbea {

class Agent {
public:
    Agent(const Config& cfg);
    void perceive(const PatternSignature& input);
    Action decide();
    void receive_reward(double reward_valence, double reward_surprise);
    void learn();

private:
    Config config;
    BeliefGraph belief_graph;
    EmotionState emotion;
    std::vector<Action> available_actions;
};

}
