#include "dbea/Agent.h"
#include "dbea/Config.h"
#include "dbea/PatternSignature.h"
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <stdexcept>
#include <string>

using namespace dbea;

// Helper: Slowly evolve baseline personality across lifetimes
void evolve_personality_baseline(EmotionState &emotion, std::mt19937 &rng)
{
    std::normal_distribution<double> noise(0.0, 0.015);
    emotion.dominance = std::clamp(emotion.dominance * 0.985 + noise(rng), 0.0, 1.0);
    emotion.fear = std::clamp(emotion.fear * 0.97 + noise(rng), 0.0, 1.0);
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

    const int num_lifetimes = 7;
    const int base_episodes = 10;
    const int therapy_episodes = 30;
    const int post_recovery_episodes = 15; // Stress test + relapse prevention

    const double exposure_prob = 0.3;
    const double therapy_trigger_prob = 0.2;

    // Post-recovery stress parameters
    const double mild_negative_prob = 0.15;      // 15% chance of small setback
    const double stress_trigger_prob = 0.35;     // higher trigger chance to test resilience
    const double negative_valence_range = -0.25; // -0.0 to -0.25

    std::string save_file = "agent_lifetime.json";

    std::cout << "DBEA v0 - Lifelong Developmental Simulation + Post-Recovery Stress & Relapse Prevention\n";

    Agent agent(cfg);

    std::ofstream emo_csv("emotion_trajectory_full.csv", std::ios::trunc);
    if (!emo_csv.is_open())
    {
        std::cerr << "Failed to open emotion_trajectory_full.csv!\n";
        return 1;
    }
    emo_csv << "Lifetime,Episode,Step,Valence,Fear,Dominance,ExploreBias,Phase\n";
    emo_csv << std::fixed << std::setprecision(6);

    std::string current_phase = "Normal";

    for (int life = 0; life < num_lifetimes; ++life)
    {
        std::cout << "\n┌────────────────────────────────────┐\n"
                  << "│ Starting Lifetime " << (life + 1) << " / " << num_lifetimes << " ";

        bool is_trauma = (life == 2);
        bool is_therapy = (life == 6);
        bool is_post_recovery = (life == 6 && true); // We'll handle phase inside therapy lifetime

        int episodes_this_life = (is_therapy) ? therapy_episodes : base_episodes;

        if (is_trauma)
        {
            std::cout << " [TRAUMA PHASE] ";
            current_phase = "Trauma";
        }
        else if (is_therapy)
        {
            std::cout << " [INTENSIVE THERAPY + POST-RECOVERY STRESS/RELAPSE] ";
            current_phase = "Therapy";
        }
        else
        {
            current_phase = "Normal";
        }

        std::cout << "│\n└────────────────────────────────────┘\n";

        if (life > 0)
        {
            try
            {
                agent.load(save_file);
                std::cout << "Loaded state from previous lifetime.\n";
                EmotionState current = agent.get_emotion();
                evolve_personality_baseline(current, rng);
                agent.set_emotion(current);
                std::cout << "Baseline → Dominance=" << current.dominance << " | Fear=" << current.fear << "\n";
            }
            catch (const std::exception &e)
            {
                std::cout << "Load failed: " << e.what() << "\n";
            }
        }

        agent.set_therapy_mode(is_therapy);

        for (int ep = 0; ep < episodes_this_life; ++ep)
        {
            std::cout << "\n=== Episode " << ep << " ===\n";
            agent.set_merge_threshold(0.95 - 0.01 * ep);

            // In therapy lifetime, after ~20 episodes → switch to stress test
            if (is_therapy && ep >= 20)
            {
                current_phase = (ep < 25) ? "Post-Recovery Stress Test" : "Relapse Prevention";
                agent.set_therapy_mode(false); // Turn off pharma boost after therapy
            }

            for (int step = 0; step < 5; ++step)
            { // steps_per_episode = 5
                int discrete_state = step % 4;
                std::uniform_real_distribution<double> small_noise(-0.01, 0.01);
                PatternSignature perception({static_cast<double>(discrete_state), 0.3 + small_noise(rng)});
                agent.perceive(perception);

                Action action = agent.decide();

                bool is_trigger = (life > 2 && discrete_state % 4 == 2);
                double trigger_p = is_therapy ? therapy_trigger_prob : stress_trigger_prob;
                is_trigger = is_trigger && (dist(rng) < trigger_p);

                double r = dist(rng);
                double reward_valence, reward_surprise = 0.05 + dist(rng) * 0.1;

                if (is_therapy)
                {
                    reward_valence = 0.1 + dist(rng) * 0.2;
                    reward_surprise = 0.05 + dist(rng) * 0.05;
                }
                else if (is_trauma)
                {
                    if (r < 0.40)
                    {
                        reward_valence = -0.45 - dist(rng) * 0.20;
                        reward_surprise += 0.7;
                    }
                    else
                    {
                        reward_valence = (r < 0.60) ? 0.06 + dist(rng) * 0.06 : 0.02 + dist(rng) * 0.02;
                    }
                }
                else
                {
                    // Normal or post-recovery
                    if (r < mild_negative_prob)
                    {
                        reward_valence = -negative_valence_range * dist(rng);
                        reward_surprise += 0.2;
                    }
                    else
                    {
                        reward_valence = 0.04 + dist(rng) * 0.08;
                    }
                }

                // Exposure only during therapy
                if (is_therapy && dist(rng) < exposure_prob && action.name == "noop")
                {
                    agent.force_action("explore");
                    action = agent.decide();
                    reward_valence = 0.4;
                    std::cout << "[DBEA] Exposure therapy: Forcing explore!\n";
                }

                if (is_trigger && agent.get_emotion().fear > 0.15)
                {
                    reward_surprise += 0.25;
                    std::cout << "[DBEA] Trigger sensitivity activated! Boosting surprise.\n";
                }

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

                std::cout << "Step " << step << " | Action: " << action.name
                          << " | Reward: " << reward_valence << " | Phase: " << current_phase << "\n";

                auto [n, e] = agent.get_proto_action_values();
                std::cout << "Proto-belief | Top: " << (e >= n ? "explore" : "noop")
                          << " | Values: (noop=" << n << ", explore=" << e << ")\n";

                emo_csv << (life + 1) << "," << ep << "," << step << ","
                        << agent.get_emotion().valence << ","
                        << agent.get_emotion().fear << ","
                        << agent.get_emotion().dominance << ","
                        << agent.get_emotion().explore_bias << ","
                        << current_phase << "\n";
            }

            agent.prune_beliefs(0.40);
        }

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

    emo_csv.close();
    std::cout << "\nSimulation complete. Full trajectory saved to emotion_trajectory_full.csv\n";
    std::cout << "Ready for GridWorld deployment in next phase.\n";
    return 0;
}