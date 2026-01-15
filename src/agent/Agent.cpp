#include "dbea/Agent.h"
#include <unordered_map>
#include <iostream>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <random>
#include <algorithm> // NEW for std::sort etc. in evo
namespace dbea
{
    Agent::Agent(const Config &cfg)
        : config(cfg), belief_graph(cfg), last_reward(0.0), rng(std::random_device{}())
    {
        available_actions.emplace_back(0, "up");
        available_actions.emplace_back(1, "down");
        available_actions.emplace_back(2, "left");
        available_actions.emplace_back(3, "right");

        auto proto = std::make_shared<BeliefNode>("proto-belief", PatternSignature({0.5, 0.5, 0.0, 0.0}));
        for (const auto &act : available_actions)
            proto->action_values[act.id] = 0.1;
        belief_graph.add_belief(proto);
        last_action = available_actions[0];

        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y)
                state_visit_count[std::to_string(x) + "_" + std::to_string(y)] = 0;
    }

    void Agent::perceive(const PatternSignature &input)
    {
        PatternSignature blended = input;
        if (!last_perception.features.empty())
        {
            for (size_t i = 0; i < blended.features.size(); ++i)
                blended.features[i] = 0.95 * blended.features[i] + 0.05 * last_perception.features[i];
        }
        last_perception = input;

        double base_threshold = 0.93;
        if (emotion.curiosity > config.curiosity_threshold)
            base_threshold -= config.curiosity_threshold_drop * emotion.curiosity;
        double dominance_effect = 0.12 * (1.0 - emotion.dominance);
        double creation_threshold = base_threshold - dominance_effect - 0.35 * emotion.curiosity;

        auto belief = belief_graph.maybe_create_belief(blended, creation_threshold);
        if (belief->action_values.empty())
        {
            for (const auto &action : available_actions)
                belief->action_values[action.id] = 0.1;
        }
        belief_graph.prune();
    }

    Action Agent::decide()
    {
        std::unordered_map<int, double> action_scores;
        total_activation = 0.0;
        for (const auto &belief : belief_graph.nodes)
            total_activation += belief->activation;

        for (const auto &belief : belief_graph.nodes)
        {
            for (const auto &action : available_actions)
            {
                double weight = belief->activation / (total_activation + 1e-6);
                action_scores[action.id] += weight * belief->predict_action_value(action.id);
            }
        }

        if (!last_perception.features.empty() && last_perception.features.size() >= 2)
        {
            double norm_x = last_perception.features[0];
            double norm_y = last_perception.features[1];
            int grid_x = static_cast<int>(norm_x * 4.999);
            int grid_y = static_cast<int>(norm_y * 4.999);
            std::string key = std::to_string(grid_x) + "_" + std::to_string(grid_y);
            state_visit_count[key]++;
            double visit_inverse = 1.0 / (1.0 + state_visit_count[key] * 0.08);
            double curiosity_bonus = config.curiosity_boost * 0.7 * visit_inverse;
            if (grid_x >= 2 && grid_y >= 2)
                curiosity_bonus *= 1.6;
            action_scores[1] += curiosity_bonus * 1.2; // down
            action_scores[3] += curiosity_bonus * 1.4; // right
        }

        static double current_epsilon = config.exploration_rate;
        current_epsilon = std::max(config.min_exploration, current_epsilon * config.epsilon_decay);

        static std::uniform_real_distribution<double> epsilon_dist(0.0, 1.0);
        if (epsilon_dist(rng) < current_epsilon)
        {
            static std::uniform_int_distribution<size_t> action_dist(0, available_actions.size() - 1);
            last_action = available_actions[action_dist(rng)];
            return last_action;
        }

        double emotional_bonus = emotion.explore_bias * config.explore_bias_scale;
        double fear_penalty = emotion.fear * 0.40;
        action_scores[3] += emotional_bonus;
        action_scores[0] -= emotional_bonus * 0.7;
        action_scores[1] -= fear_penalty;

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
        emotion.update(valence, surprise, 0.0, config);
        last_reward = valence;
    }

    void Agent::learn()
    {
        double total_error = 0.0;
        int count = 0;
        double reinforcement_mod = 1.0 + 0.5 * emotion.valence + 0.3 * emotion.arousal;
        double fear_decay_boost = 1.0 + 0.2 * emotion.fear;

        double progress_bonus = 0.0;
        if (!last_perception.features.empty() && last_perception.features.size() >= 2)
        {
            double norm_x = last_perception.features[0];
            double norm_y = last_perception.features[1];
            progress_bonus = (norm_x * 0.12) + (norm_y * 0.18);
        }

        // NEW: Track co-activations for symbiosis
        std::vector<std::shared_ptr<BeliefNode>> active_beliefs;
        for (auto &belief : belief_graph.nodes)
        {
            if (belief->activation > config.co_activation_thresh)
                active_beliefs.push_back(belief);
        }
        for (size_t i = 0; i < active_beliefs.size(); ++i)
        {
            for (size_t j = i + 1; j < active_beliefs.size(); ++j)
            {
                std::string key = std::min(active_beliefs[i]->id, active_beliefs[j]->id) + "_" +
                                  std::max(active_beliefs[i]->id, active_beliefs[j]->id);
                belief_graph.co_activations[key]++;
            }
        }

        // Enhanced fitness update
        double avg_fitness = 0.0;
        for (auto &belief : belief_graph.nodes)
            avg_fitness += belief->fitness;
        avg_fitness /= (belief_graph.nodes.size() + 1e-6);

        for (auto &belief : belief_graph.nodes)
        {
            double credit = belief->activation * (last_reward + progress_bonus);
            double surprise_factor = 1.0 + 2.5 * std::abs(last_reward - last_predicted_reward);
            belief->learn_action_value(last_action.id, credit, belief->local_lr * surprise_factor, config.gamma);

            if (credit > 0.0)
                belief->reinforce(config.belief_learning_rate * credit * reinforcement_mod);
            else
                belief->decay(config.belief_decay_rate * fear_decay_boost);

            double error = std::abs(last_reward - last_predicted_reward);
            belief->prediction_error = 0.7 * belief->prediction_error + 0.3 * error;
            total_error += belief->prediction_error * belief->activation;
            count++;

            // NEW: Fitness update with niche and symbiotic terms
            double alpha_bt = belief->activation / (total_activation + 1e-6);
            double delta_td = last_reward + config.gamma * belief->predict_action_value(last_action.id) - belief->last_predicted_reward;
            bool causal_mask = (alpha_bt > 0.2); // Simplified mask
            double regret_bonus = std::max(0.0, std::max({belief->action_values[0], belief->action_values[1], belief->action_values[2], belief->action_values[3]}) - belief->action_values[last_action.id]);
            double regret_penalty = std::max(0.0, belief->action_values[last_action.id] - std::max({belief->action_values[0], belief->action_values[1], belief->action_values[2], belief->action_values[3]}));

            // Niche density: Count close beliefs
            int niche_count = 0;
            for (const auto &other : belief_graph.nodes)
                if (other != belief && belief->match_score(other->prototype) > config.niche_radius)
                    niche_count++;
            double niche_density = niche_count / (belief_graph.nodes.size() + 1e-6);
            double niche_bonus = config.niche_bonus_scale * (1.0 - niche_density);

            // Symbiotic uplift
            double symbiotic_income = 0.0;
            int partner_count = 0;
            for (const auto &[key, count] : belief_graph.co_activations)
            {
                if (key.find(belief->id) != std::string::npos && count > 5) // Arbitrary min co-act
                {
                    // Parse other ID from key
                    std::string other_id = key.substr(0, key.find("_")) == belief->id ? key.substr(key.find("_") + 1) : key.substr(0, key.find("_"));
                    for (const auto &other : belief_graph.nodes)
                    {
                        if (other->id == other_id)
                        {
                            symbiotic_income += other->fitness;
                            partner_count++;
                            break;
                        }
                    }
                }
            }
            double symbiotic_uplift_term = config.symbiotic_uplift * (symbiotic_income / (partner_count + 1e-6));

            // Delta fitness
            double delta_fitness = 0.01 * alpha_bt * delta_td * (causal_mask ? 1.0 : 0.0) + 0.2 * regret_bonus - 0.4 * regret_penalty + niche_bonus + symbiotic_uplift_term;

            belief->fitness += delta_fitness;
        }

        double avg_error = (count > 0) ? total_error / count : 0.0;
        emotion.update(last_reward, 0.05, avg_error, config);

        if (!belief_graph.nodes.empty())
        {
            auto &proto = *belief_graph.nodes.front();
            if (proto.id == "proto-belief")
                proto.decay(0.035);
        }

        double dynamic_merge = config.merge_threshold + 0.06 * (1.0 - emotion.dominance);
        belief_graph.merge_beliefs(dynamic_merge);

        // Adaptive prune: much gentler when population is low
        double prune_thresh;
        size_t current_size = belief_graph.nodes.size();
        if (current_size < 5)
            prune_thresh = 0.05;           // Very gentle — almost no pruning
        else if (current_size < config.min_beliefs_before_prune)
            prune_thresh = 0.12;
        else
            prune_thresh = 0.32;

        prune_beliefs(prune_thresh);

        // NEW: Evolutionary cycle trigger
        static int step_count = 0;
        static double reward_buffer = 0.0;
        static int low_reward_streak = 0;
        step_count++;
        reward_buffer = 0.95 * reward_buffer + 0.05 * last_reward;
        if (last_reward < 0.0)
            low_reward_streak++;
        else
            low_reward_streak = 0;

        // Emotional scaling: Freq ∝ arousal
        int effective_freq = static_cast<int>(config.evo_cycle_freq / (1.0 + emotion.arousal));

        if (step_count % effective_freq == 0 || reward_buffer < config.crisis_reward_thresh || low_reward_streak > 5)
        {
            belief_graph.evolve_cycle(emotion);
            step_count = 0; // Optional reset
        }

        // Debug (unchanged)
        std::cout << "[DBEA] === Step Summary ===\n";
        for (const auto &belief : belief_graph.nodes)
        {
            std::cout << "[DBEA] " << belief->id
                      << " conf=" << belief->confidence
                      << " fitness=" << belief->fitness
                      << " mut_rate=" << belief->mutation_rate
                      << " values: ";
            for (const auto &[id, v] : belief->action_values)
                std::cout << "(" << id << ":" << v << ") ";
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

    // Serialization updates: Add new fields
    json Agent::to_json() const
    {
        json j;
        j["emotion"]["valence"] = emotion.valence;
        j["emotion"]["arousal"] = emotion.arousal;
        j["emotion"]["dominance"] = emotion.dominance;
        j["emotion"]["curiosity"] = emotion.curiosity;
        j["emotion"]["fear"] = emotion.fear;
        j["emotion"]["explore_bias"] = emotion.explore_bias;

        json beliefs_arr = json::array();
        for (const auto &node : belief_graph.nodes)
        {
            json b;
            b["id"] = node->id;
            b["confidence"] = node->confidence;
            b["evidence_count"] = node->evidence_count;
            b["prototype"] = node->prototype.features;
            b["fitness"] = node->fitness;                       // NEW
            b["mutation_rate"] = node->mutation_rate;           // NEW
            b["local_lr"] = node->local_lr;                     // NEW
            b["emotional_affinity"] = node->emotional_affinity; // NEW
            json action_vals = json::object();
            for (const auto &[id, val] : node->action_values)
                action_vals[std::to_string(id)] = val;
            b["action_values"] = action_vals;
            beliefs_arr.push_back(b);
        }
        j["beliefs"] = beliefs_arr;

        // NEW: Save co_activations
        json co_act = json::object();
        for (const auto &[key, val] : belief_graph.co_activations)
            co_act[key] = val;
        j["co_activations"] = co_act;

        return j;
    }

    void Agent::from_json(const json &j)
    {
        belief_graph.nodes.clear();
        belief_graph.co_activations.clear(); // NEW

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

        if (j.contains("beliefs"))
        {
            for (const auto &b : j["beliefs"])
            {
                std::string id = b["id"];
                std::vector<double> proto_features = b["prototype"].get<std::vector<double>>();
                auto node = std::make_shared<BeliefNode>(id, PatternSignature(proto_features));
                node->confidence = b.value("confidence", 0.5);
                node->evidence_count = b.value("evidence_count", 1);
                node->fitness = b.value("fitness", 0.0);                                               // NEW
                node->mutation_rate = b.value("mutation_rate", 0.1);                                   // NEW
                node->local_lr = b.value("local_lr", config.learning_rate);                            // NEW
                node->emotional_affinity = b.value("emotional_affinity", std::vector<double>(5, 0.0)); // NEW
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

        // NEW: Load co_activations
        if (j.contains("co_activations"))
        {
            for (auto &[key, val] : j["co_activations"].items())
                belief_graph.co_activations[key] = val.get<int>();
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
