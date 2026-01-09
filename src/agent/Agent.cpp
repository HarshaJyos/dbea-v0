#include "dbea/Agent.h"
#include <memory>

namespace dbea {

Agent::Agent(const Config& cfg) : config(cfg) {
    available_actions.push_back(Action{0, "noop"});
    available_actions.push_back(Action{1, "explore"});

    belief_graph.add_belief(
        std::make_shared<BeliefNode>(
            "proto-belief",
            PatternSignature({0.1, 0.1})
        )
    );
}

void Agent::perceive(const PatternSignature& input) {
    belief_graph.maybe_create_belief(input, 0.8);
    belief_graph.prune();
}


Action Agent::decide() {
    return available_actions[0];
}

void Agent::receive_reward(double v, double s) {
    emotion.update(v, s);
}

void Agent::learn() {}

}
