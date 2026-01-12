#pragma once
#include "dbea/BeliefGraph.h"
#include "dbea/EmotionState.h"
#include "dbea/Environment.h"
#include "dbea/Action.h"
#include "dbea/PatternSignature.h"
#include "dbea/Config.h"
#include <vector>
#include <utility>
#include <nlohmann/json.hpp>
#include <random>           // ← ADD THIS LINE
using json = nlohmann::json;
namespace dbea
{
    class Agent
    {
    public:
        Agent(const Config &cfg);
        void perceive(const PatternSignature &input);
        Action decide();
        void receive_reward(double reward_valence, double reward_surprise);
        void learn();
        // Public methods (make prune public, not inline)
        void prune_beliefs(double threshold = 0.40);
        std::pair<double, double> get_proto_action_values() const;
        size_t get_belief_count() const;
        std::vector<std::pair<double, double>> get_all_belief_action_values() const;
        const EmotionState &get_emotion() const { return emotion; }
        void set_emotion(const EmotionState &new_emotion) { emotion = new_emotion; }
        // Serialization
        json to_json() const;
        void from_json(const json &j);
        void save(const std::string &filename) const;
        void load(const std::string &filename);
        void set_therapy_mode(bool enabled);
        void set_merge_threshold(double threshold);
        void force_action(const std::string &action_name);

    private:
        Config config;
        BeliefGraph belief_graph;
        EmotionState emotion;
        std::vector<Action> available_actions;
        double last_reward = 0.0;
        Action last_action;
        PatternSignature last_perception;
        double last_predicted_reward = 0.0;
        std::mt19937 rng;  // ← Add this random engine
    };
} // namespace dbea