#include "dbea/Agent.h"
#include <unordered_map>
#include <iostream>
#include <cmath>     // for std::abs
#include <fstream>   // NEW
#include <stdexcept> // NEW for exceptions
namespace dbea
{
    Agent::Agent(const Config &cfg)
        : config(cfg),
          belief_graph(cfg),
          last_reward(0.0)
    {
        available_actions.push_back(Action{0, "noop"});
        available_actions.push_back(Action{1, "explore"});
        auto proto = std::make_shared<BeliefNode>("proto-belief", PatternSignature({0.1, 0.1}));
        proto->action_values[0] = 0.1;
        proto->action_values[1] = 0.2;
        belief_graph.add_belief(proto);
        last_action = available_actions[0];
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
        // Base threshold
        double base_threshold = 0.93;
        // Curiosity effect
        if (emotion.curiosity > config.curiosity_threshold)
        {
            base_threshold -= config.curiosity_threshold_drop * emotion.curiosity;
            std::cout << "[DBEA] Curiosity active (" << emotion.curiosity
                      << ") → threshold=" << base_threshold << "\n";
        }
        // NEW: Low dominance → lower threshold → easier to create new beliefs
        double dominance_effect = 0.12 * (1.0 - emotion.dominance); // 0 → 0.12 range
        double creation_threshold = base_threshold - dominance_effect;
        // Create or match belief using the final threshold
        auto belief = belief_graph.maybe_create_belief(blended, creation_threshold);
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
        // Base Q-values from beliefs
        for (const auto &belief : belief_graph.nodes)
        {
            for (const auto &action : available_actions)
            {
                double weight = belief->activation / (total_activation + 1e-6);
                action_scores[action.id] += weight * belief->predict_action_value(action.id);
            }
        }
        // NEW: Apply emotional bias to action scores
        double emotional_bonus = emotion.explore_bias * 0.25; // was 0.15
        double fear_penalty = emotion.fear * 0.25;
        action_scores[1] -= fear_penalty;    // extra penalty to explore when afraid
        action_scores[1] += emotional_bonus; // boost explore (id=1)
        action_scores[0] -= emotional_bonus; // reduce noop (id=0)
        // Select best action
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
        // Predict reward for chosen action (for next error calculation)
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
        emotion.update(valence, surprise, 0.0, config); // avg_error comes later in learn()
        last_reward = valence;
    }
    void Agent::learn()
    {
        double total_error = 0.0;
        int count = 0;
        // NEW: Emotional modulation factors
        double reinforcement_mod = 1.0 + 0.5 * emotion.valence + 0.3 * emotion.arousal;
        double fear_decay_boost = 1.0 + 0.2 * emotion.fear; // fear → stronger decay
        for (auto &belief : belief_graph.nodes)
        {
            double credit = belief->activation * last_reward;
            belief->learn_action_value(last_action.id, credit, config.learning_rate);
            if (credit > 0.0)
            {
                // Positive credit → reinforce stronger when valence/arousal high
                belief->reinforce(config.belief_learning_rate * credit * reinforcement_mod);
            }
            else
            {
                // Negative → decay stronger when fear is high
                belief->decay(config.belief_decay_rate * fear_decay_boost);
            }
            // Update prediction error
            double error = std::abs(last_reward - last_predicted_reward);
            belief->prediction_error = 0.7 * belief->prediction_error + 0.3 * error;
            total_error += belief->prediction_error * belief->activation;
            count++;
        }
        double avg_error = (count > 0) ? total_error / count : 0.0;
        // Update emotions with computed avg_error
        emotion.update(last_reward, 0.05, avg_error, config);
        if (!belief_graph.nodes.empty())
        {
            auto &proto = *belief_graph.nodes.front();
            if (proto.id == "proto-belief")
            {
                proto.decay(0.04);
            }
        }
        double dynamic_merge_threshold = config.merge_threshold + 0.08 * (1.0 - emotion.dominance);
        belief_graph.merge_beliefs(dynamic_merge_threshold);
        // Debug output - cleaner version
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
                  << " | Curiosity: " << emotion.curiosity
                  << " | Valence: " << emotion.valence
                  << " | Fear: " << emotion.fear
                  << " | Dominance: " << emotion.dominance
                  << " | Explore bias: " << emotion.explore_bias << "\n"
                  << "[DBEA] Belief count: " << belief_graph.nodes.size() << "\n\n";
    }
    // ──────────────────────────────────────────────────────────────────────────────
    // Serialization / Persistence for multi-lifetime learning
    // ──────────────────────────────────────────────────────────────────────────────
    json Agent::to_json() const
    {
        json j;
        // Emotions (core of lifelong personality)
        j["emotion"]["valence"] = emotion.valence;
        j["emotion"]["arousal"] = emotion.arousal;
        j["emotion"]["dominance"] = emotion.dominance;
        j["emotion"]["curiosity"] = emotion.curiosity;
        j["emotion"]["fear"] = emotion.fear;
        j["emotion"]["explore_bias"] = emotion.explore_bias;
        // Beliefs
        json beliefs_arr = json::array();
        for (const auto &node : belief_graph.nodes)
        {
            json b;
            b["id"] = node->id;
            b["confidence"] = node->confidence;
            b["evidence_count"] = node->evidence_count;
            b["prototype"] = node->prototype.features;
            json action_vals = json::object();
            for (const auto &[id, val] : node->action_values)
            {
                action_vals[std::to_string(id)] = val;
            }
            b["action_values"] = action_vals;
            beliefs_arr.push_back(b);
        }
        j["beliefs"] = beliefs_arr;
        return j;
    }
    void Agent::from_json(const json &j)
    {
        // Clear current state
        belief_graph.nodes.clear();
        // Load emotions
        if (j.contains("emotion"))
        {
            auto &e = j["emotion"];
            emotion.valence = e.value("valence", 0.0);
            emotion.arousal = e.value("arousal", 0.0);
            emotion.dominance = e.value("dominance", 0.5);
            emotion.curiosity = e.value("curiosity", 0.5);
            emotion.fear = e.value("fear", 0.0);
            emotion.explore_bias = e.value("explore_bias", 0.0);
        }
        // Load beliefs
        if (j.contains("beliefs"))
        {
            for (const auto &b : j["beliefs"])
            {
                std::string id = b["id"];
                std::vector<double> proto_features = b["prototype"].get<std::vector<double>>();
                auto node = std::make_shared<BeliefNode>(id, PatternSignature(proto_features));
                node->confidence = b.value("confidence", 0.5);
                node->evidence_count = b.value("evidence_count", 1);
                if (b.contains("action_values"))
                {
                    for (auto &[key, val] : b["action_values"].items())
                    {
                        int act_id = std::stoi(key);
                        node->action_values[act_id] = val.get<double>();
                    }
                }
                belief_graph.add_belief(node);
            }
        }
    }
    void Agent::save(const std::string &filename) const
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open save file: " + filename);
        }
        file << to_json().dump(2);
        file.close();
    }
    void Agent::load(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open load file: " + filename);
        }
        json j;
        file >> j;
        from_json(j);
        file.close();
    }
    // NEW: Define the missing methods
    std::pair<double, double> Agent::get_proto_action_values() const
    {
        if (belief_graph.nodes.empty())
            return {0.0, 0.0};
        const auto &proto = belief_graph.nodes.front();
        return {proto->predict_action_value(0),
                proto->predict_action_value(1)};
    }
    size_t Agent::get_belief_count() const
    {
        return belief_graph.nodes.size();
    }
    std::vector<std::pair<double, double>> Agent::get_all_belief_action_values() const
    {
        std::vector<std::pair<double, double>> values;
        for (const auto &belief : belief_graph.nodes)
        {
            values.emplace_back(
                belief->predict_action_value(0),
                belief->predict_action_value(1));
        }
        return values;
    }
    void Agent::prune_beliefs(double threshold)
    {
        belief_graph.prune(threshold);
    }
    void Agent::set_therapy_mode(bool enabled)
    {
        config.therapy_mode = enabled;
    }

    void Agent::set_merge_threshold(double threshold)
    {
        config.merge_threshold = threshold;
    }

    void Agent::force_action(const std::string &action_name)
    {
        for (const auto &act : available_actions)
        {
            if (act.name == action_name)
            {
                last_action = act;
                std::cout << "[DBEA] Forced action: " << action_name << "\n";
                return;
            }
        }
        std::cerr << "[WARNING] Could not force action: " << action_name << " not found!\n";
    }
} // namespace dbea