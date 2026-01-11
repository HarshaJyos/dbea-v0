#include "dbea/Agent.h"
#include <unordered_map>
#include <iostream>
#include <cmath> // for std::abs

namespace dbea
{

    Agent::Agent(const Config &cfg)
        : config(cfg),
          belief_graph(cfg),
          last_reward(0.0)
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
            }
        }
        last_perception = input; // store raw

        double threshold = 0.93;
        if (emotion.curiosity > config.curiosity_threshold)
        {
            threshold -= config.curiosity_threshold_drop * emotion.curiosity;
            std::cout << "[DBEA] Curiosity active (" << emotion.curiosity
                      << ") → threshold=" << threshold << "\n";
        }

        auto belief = belief_graph.maybe_create_belief(blended, threshold); // FIXED: use blended

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

        double total_activation = 0.0;
        for (const auto &belief : belief_graph.nodes)
        {
            total_activation += belief->activation;
        }

        double predicted = 0.0;
        for (const auto &belief : belief_graph.nodes)
        {
            for (const auto &action : available_actions)
            {
                double weight = belief->activation / (total_activation + 1e-6);
                action_scores[action.id] += weight * belief->predict_action_value(action.id);
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

        // FIXED: Predict for the CHOSEN action (after selection)
        double chosen_pred = 0.0;
        for (const auto &belief : belief_graph.nodes)
        {
            double weight = belief->activation / (total_activation + 1e-6);
            chosen_pred += weight * belief->predict_action_value(best.id);
        }
        last_predicted_reward = chosen_pred;

        return best;
    }

    void Agent::receive_reward(double valence, double surprise)
    {
        emotion.update(valence, surprise, 0.0, config); // safe default avg_error=0
        last_reward = valence;
    }

    void Agent::learn()
    {
        double total_error = 0.0;
        int count = 0;

        for (auto &belief : belief_graph.nodes)
        {
            double credit = belief->activation * last_reward;

            belief->learn_action_value(last_action.id, credit, config.learning_rate);

            if (credit > 0.0)
            {
                belief->reinforce(config.belief_learning_rate * credit);
            }
            else
            {
                belief->decay(config.belief_decay_rate);
            }

            // Update prediction error
            double error = std::abs(last_reward - last_predicted_reward);
            belief->prediction_error = 0.7 * belief->prediction_error + 0.3 * error; // EMA
            total_error += belief->prediction_error * belief->activation;
            count++;
        }

        double avg_error = (count > 0) ? total_error / count : 0.0;

        // FIXED: Call 3-param update AFTER computing avg_error
        // In learn():
        emotion.update(last_reward, 0.05, avg_error, config); // ← Add config
        if (!belief_graph.nodes.empty())
        {
            auto &proto = *belief_graph.nodes.front();
            if (proto.id == "proto-belief")
            {
                proto.decay(0.04);
            }
        }

        // FIXED: Remove duplicate
        belief_graph.merge_beliefs(config.merge_threshold);

        // Debug output
        // Debug output (once per step)
        std::cout << "[DBEA] === Step Summary ===\n";
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
        std::cout << "[DBEA] Avg prediction error: " << avg_error
                  << " | Curiosity: " << emotion.curiosity << "\n"
                  << "[DBEA] Belief count: " << belief_graph.nodes.size() << "\n\n";
    }

} // namespace dbea