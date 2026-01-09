#include "dbea/Agent.h"
#include <memory>
#include <unordered_map>
#include <iostream>

namespace dbea {

Agent::Agent(const Config& cfg) : config(cfg), last_reward(0.0) {
    // Define available actions
    available_actions.push_back(Action{0, "noop"});
    available_actions.push_back(Action{1, "explore"});

    // Seed with proto-belief (initial naive belief)
    auto proto = std::make_shared<BeliefNode>(
        "proto-belief",
        PatternSignature({0.1, 0.1})
    );

    // Initialize expected action values for proto-belief
    proto->action_values[0] = 0.1; // noop
    proto->action_values[1] = 0.2; // explore

    belief_graph.add_belief(proto);
}

void Agent::perceive(const PatternSignature& input) {
    // Possibly create a new belief if activation is low
    auto belief = belief_graph.maybe_create_belief(input, 0.8);

    // Initialize newborn belief with neutral expectations
    if (belief->action_values.empty()) {
        for (const auto& action : available_actions) {
            belief->action_values[action.id] = 0.1;
        }
    }

    // Prune low-confidence beliefs
    belief_graph.prune();
}

Action Agent::decide() {
    // Aggregate votes from all beliefs
    std::unordered_map<int, double> action_scores;

    for (const auto& belief : belief_graph.nodes) {
        for (const auto& action : available_actions) {
            double vote = belief->activation *
                          belief->predict_action_value(action.id);
            action_scores[action.id] += vote;
        }
    }

    // Pick the best action
    Action best = available_actions[0];
    double best_score = -1e9;

    for (const auto& action : available_actions) {
        double score = action_scores[action.id];
        if (score > best_score) {
            best_score = score;
            best = action;
        }
    }

    return best;
}

void Agent::receive_reward(double valence, double surprise) {
    // Update agent’s emotional state
    emotion.update(valence, surprise);

    // Store last reward for learning
    last_reward = valence; // simple model: use valence as reward
}

void Agent::learn() {
    // Update beliefs' action values based on last reward
    for (auto& belief : belief_graph.nodes) {
        for (const auto& action : available_actions) {
            double scaled_reward = belief->activation * last_reward;
            belief->learn_action_value(action.id, scaled_reward, config.learning_rate);
        }
    }

    // Optional: log updated beliefs
    for (const auto& belief : belief_graph.nodes) {
        std::cout << "[DBEA] " << belief->id << " action values: ";
        for (const auto& [action_id, value] : belief->action_values) {
            std::cout << "(" << action_id << ": " << value << ") ";
        }
        std::cout << std::endl;
    }
}

} // namespace dbea
