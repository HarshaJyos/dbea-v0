#include "dbea/Agent.h"
#include <memory>
#include <unordered_map>
#include <iostream>

namespace dbea
{

    Agent::Agent(const Config &cfg) : config(cfg), last_reward(0.0)
    {
        available_actions.push_back(Action{0, "noop"});
        available_actions.push_back(Action{1, "explore"});

        auto proto = std::make_shared<BeliefNode>(
            "proto-belief",
            PatternSignature({0.1, 0.1}));

        proto->action_values[0] = 0.1;
        proto->action_values[1] = 0.2;

        belief_graph.add_belief(proto);

        // Initialize last_action safely
        last_action = available_actions[0];
    }

    void Agent::perceive(const PatternSignature &input)
    {
        auto belief = belief_graph.maybe_create_belief(input, 0.8);

        if (belief->action_values.empty())
        {
            for (const auto &action : available_actions)
            {
                belief->action_values[action.id] = 0.1;
            }
        }

        belief_graph.prune();
    }

    Action Agent::decide()
    {
        std::unordered_map<int, double> action_scores;

        for (const auto &belief : belief_graph.nodes)
        {
            for (const auto &action : available_actions)
            {
                double vote =
                    belief->activation *
                    belief->predict_action_value(action.id);

                action_scores[action.id] += vote;
            }
        }

        Action best = available_actions[0];
        double best_score = -1e9;

        for (const auto &action : available_actions)
        {
            double score = action_scores[action.id];
            if (score > best_score)
            {
                best_score = score;
                best = action;
            }
        }

        // ✅ STORE LAST ACTION
        last_action = best;
        return best;
    }

    void Agent::receive_reward(double valence, double surprise)
    {
        emotion.update(valence, surprise);
        last_reward = valence;
    }

    void Agent::learn()
    {
        for (auto &belief : belief_graph.nodes)
        {
            double scaled_reward = belief->activation * last_reward;

            // ✅ Learn ONLY the chosen action
            belief->learn_action_value(
                last_action.id,
                scaled_reward,
                config.learning_rate);
        }

        for (const auto &belief : belief_graph.nodes)
        {
            std::cout << "[DBEA] " << belief->id << " action values: ";
            for (const auto &[action_id, value] : belief->action_values)
            {
                std::cout << "(" << action_id << ": " << value << ") ";
            }
            std::cout << std::endl;
        }
        for (auto &belief : belief_graph.nodes)
        {
            double credit = belief->activation * last_reward;

            // Strengthen belief confidence
            belief->reinforce(config.belief_learning_rate * credit);

            // Optional decay for inactive beliefs
            belief->decay(config.belief_decay_rate);
        }
    }

} // namespace dbea
