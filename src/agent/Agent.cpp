#include "dbea/Agent.h"

namespace dbea {

Agent::Agent(const Config& cfg) : config(cfg) {
    available_actions.push_back(Action{0, "noop", 0.0, 0.0, 0.0});
    available_actions.push_back(Action{1, "action1", 1.0, 0.1, 0.2});
}

Action Agent::decide() {
    return available_actions[0];
}

void Agent::perceive(const PatternSignature&) {}
void Agent::receive_reward(double v, double s) { emotion.update(v, s); }
void Agent::learn() {}

}
