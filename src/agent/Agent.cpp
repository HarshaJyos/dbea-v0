#include "dbea/Agent.h"
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

        last_action = available_actions[0]; // Safe init
    }

    void Agent::perceive(const PatternSignature &input)
    {
        PatternSignature blended = input;
        if (!last_perception.features.empty())
        {
            for (size_t i = 0; i < blended.features.size(); ++i)
            {
                blended.features[i] = 0.95 * blended.features[i] + 0.05 * last_perception.features[i];
                // ^^^ much stronger persistence
            }
        }
        last_perception = input; // still store raw for next blend
        // Raised threshold to 0.95 to reduce excessive belief creation
        auto belief = belief_graph.maybe_create_belief(input, 0.75);

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
                action_scores[action.id] +=
                    belief->activation *
                    belief->predict_action_value(action.id);
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
            double credit = belief->activation * last_reward;

            // Action learning (ONLY chosen action)
            belief->learn_action_value(
                last_action.id,
                credit,
                config.learning_rate);

            // Belief confidence learning (no duplicate calls)
            if (credit > 0.0)
                belief->reinforce(config.belief_learning_rate * credit);
            else
                belief->decay(config.belief_decay_rate);
        }
        if (!belief_graph.nodes.empty())
        {
            auto &proto = *belief_graph.nodes.front();
            if (proto.id == "proto-belief")
            {
                proto.decay(0.04); // extra decay for proto so it weakens over time
            }
        }

        // Debug output (with confidence)
        for (const auto &belief : belief_graph.nodes)
        {
            std::cout << "[DBEA] " << belief->id
                      << " conf=" << belief->confidence
                      << " values: ";
            for (const auto &[id, v] : belief->action_values)
            {
                std::cout << "(" << id << ":" << v << ") ";
            }
            std::cout << std::endl;
        }
    }

} // namespace dbea