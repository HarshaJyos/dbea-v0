#include "dbea/Agent.h"
#include "dbea/Config.h"
#include "dbea/PatternSignature.h"
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <stdexcept>

using namespace dbea;

// Helper: Slowly evolve baseline personality across lifetimes
// Helper: Slowly evolve baseline personality across lifetimes
void evolve_personality_baseline(EmotionState &emotion, std::mt19937 &rng)
{
    std::normal_distribution<double> noise(0.0, 0.015);
    emotion.dominance = std::clamp(emotion.dominance * 0.985 + noise(rng), 0.0, 1.0);
    emotion.fear = std::clamp(emotion.fear * 0.97 + noise(rng), 0.0, 1.0);
    // Reset transient emotions each lifetime
    emotion.valence = 0.0;
    emotion.curiosity = 0.5;
    emotion.explore_bias = 0.0;
}

int main()
{
    Config cfg;
    cfg.max_beliefs = 100;
    cfg.exploration_rate = 0.8;
    cfg.learning_rate = 0.1;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    const int num_lifetimes = 6; // 6 lives total
    const int episodes_per_lifetime = 10;
    const int steps_per_episode = 5;

    std::string save_file = "agent_lifetime.json";

    std::cout << "DBEA v0 - Lifelong Developmental Simulation (6 lifetimes)\n";

    Agent agent(cfg);

    for (int life = 0; life < num_lifetimes; ++life)
    {
        std::cout << "\n┌────────────────────────────────────┐\n"
                  << "│ Starting Lifetime " << (life + 1) << " / " << num_lifetimes << "   ";

        bool is_trauma = (life == 2);
        if (is_trauma)
        {
            std::cout << " [TRAUMA PHASE] ";
        }
        std::cout << "│\n└────────────────────────────────────┘\n";

        // Load from previous life (skip first)
        if (life > 0)
        {
            try
            {
                agent.load(save_file);
                std::cout << "Loaded state from previous lifetime.\n";

                // Evolve baseline personality (using getter + setter)
                EmotionState current_emotion = agent.get_emotion();
                evolve_personality_baseline(current_emotion, rng);
                agent.set_emotion(current_emotion);

                std::cout << "Current baseline: Dominance=" << current_emotion.dominance
                          << " | Fear=" << current_emotion.fear << "\n";
            }
            catch (const std::exception &e)
            {
                std::cout << "No previous state or load failed: " << e.what() << "\n";
            }
        }

        for (int ep = 0; ep < episodes_per_lifetime; ++ep)
        {
            std::cout << "\n=== Episode " << ep << " ===\n";
            cfg.merge_threshold = 0.95 - 0.01 * ep;

            for (int step = 0; step < steps_per_episode; ++step)
            {
                // Perception
                int discrete_state = step % 4;
                std::uniform_real_distribution<double> small_noise(-0.01, 0.01);
                PatternSignature perception({static_cast<double>(discrete_state), 0.3 + small_noise(rng)});
                agent.perceive(perception);

                Action action = agent.decide();

                // Reward logic (trauma = much harsher negatives)
                double r = dist(rng);
                double reward_valence, reward_surprise = 0.05 + (dist(rng) * 0.1);

                if (is_trauma)
                {
                    // Trauma phase: 30% chance severe negative, higher punishment
                    if (r < 0.30)
                    {
                        reward_valence = -0.35 - (dist(rng) * 0.15); // -0.35 to -0.5
                        reward_surprise += 0.6;
                    }
                    else if (r < 0.50)
                    {
                        reward_valence = 0.08 + (dist(rng) * 0.08);
                    }
                    else
                    {
                        reward_valence = 0.02 + (dist(rng) * 0.03);
                    }
                }
                else
                {
                    // Normal phase
                    if (r < 0.08)
                    {
                        reward_valence = -0.15 - (dist(rng) * 0.1);
                        reward_surprise += 0.4;
                    }
                    else if (r < 0.38)
                    {
                        reward_valence = 0.15 + (dist(rng) * 0.1);
                    }
                    else
                    {
                        reward_valence = 0.04 + (dist(rng) * 0.04);
                    }
                }

                // Streak punishment (same for both)
                static int explore_streak = 0;
                if (action.name == "explore")
                {
                    explore_streak++;
                    if (explore_streak >= cfg.max_explore_streak && dist(rng) < cfg.streak_punish_prob)
                    {
                        reward_valence += cfg.streak_punish_amount;
                    }
                }
                else
                {
                    explore_streak = 0;
                }

                agent.receive_reward(reward_valence, reward_surprise);
                agent.learn();

                // Logging...
                std::cout << "Step " << step << " | Action: " << action.name
                          << " | Reward: " << reward_valence << "\n";

                auto [n, e] = agent.get_proto_action_values();
                std::cout << "Proto-belief | Top Action: " << (e >= n ? "explore" : "noop")
                          << " | Values: (noop=" << n << ", explore=" << e << ")\n";

                // Inside step loop, after agent.learn()
                static std::ofstream emo_csv("emotion_trajectory.csv", std::ios::app);
                if (!emo_csv.is_open())
                    emo_csv.open("emotion_trajectory.csv");
                if (life == 0 && ep == 0 && step == 0)
                {
                    emo_csv << "Lifetime,Episode,Step,Valence,Fear,Dominance,ExploreBias\n";
                }
                emo_csv << (life + 1) << "," << ep << "," << step << ","
                        << agent.get_emotion().valence << ","
                        << agent.get_emotion().fear << ","
                        << agent.get_emotion().dominance << ","
                        << agent.get_emotion().explore_bias << "\n";
                emo_csv.flush();
            }

            agent.prune_beliefs(0.40);
        }

        // Save at end of lifetime
        try
        {
            agent.save(save_file);
            std::cout << "Saved state after lifetime " << (life + 1) << "\n";
        }
        catch (const std::exception &e)
        {
            std::cout << "Save failed: " << e.what() << "\n";
        }
    }

    std::cout << "\nAll lifetimes completed.\n";
    return 0;
}