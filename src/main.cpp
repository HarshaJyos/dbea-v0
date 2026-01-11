#include "dbea/Agent.h"
#include "dbea/Config.h"
#include "dbea/PatternSignature.h"
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>

using namespace dbea;

int main()
{
    Config cfg;
    cfg.max_beliefs = 100;
    cfg.exploration_rate = 0.8;
    cfg.learning_rate = 0.1;

    Agent agent(cfg);

    std::cout << "DBEA v0 learning simulation running." << std::endl;

    const int episodes = 10;
    const int steps_per_episode = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> small_noise(-0.01, 0.01);

    std::ofstream csv("learning_curve.csv");
    csv << "Episode,Step,Action_noop,Action_explore\n";
    csv << std::fixed << std::setprecision(6);

    for (int ep = 0; ep < episodes; ++ep)
    {
        std::cout << "\n=== Episode " << ep << " ===" << std::endl;

        for (int step = 0; step < steps_per_episode; ++step)
        {
            int discrete_state = step % 4;
            PatternSignature perception({
                static_cast<double>(discrete_state),
                0.3 + small_noise(rng)
            });

            agent.perceive(perception);

            Action action = agent.decide();

            double reward_valence = 0.05 + 0.15 * ((action.name == "explore") && (step % 2 == 0));
            double reward_surprise = 0.05;

            agent.receive_reward(reward_valence, reward_surprise);
            agent.learn();

            std::cout << "Step " << step
                      << " | Action: " << action.name
                      << " | Reward: " << reward_valence << std::endl;

            auto [noop_val, explore_val] = agent.get_proto_action_values();
            csv << ep << "," << step << "," << noop_val << "," << explore_val << "\n";

            std::string top_action = (explore_val >= noop_val) ? "explore" : "noop";

            std::cout << "Proto-belief"
                      << " | Top Action: " << top_action
                      << " | Values: (noop=" << noop_val
                      << ", explore=" << explore_val << ")\n";
        }

        // Prune once after full episode
        agent.prune_beliefs(0.40);

        std::cout << "\n[Episode " << ep << " Summary]\n";
        auto all_values = agent.get_all_belief_action_values();
        for (size_t i = 0; i < all_values.size(); ++i)
        {
            auto [noop_val, explore_val] = all_values[i];
            std::string top = (explore_val >= noop_val) ? "explore" : "noop";
            std::cout << "Belief " << i
                      << " | Top Action: " << top
                      << " | Values: (noop=" << noop_val
                      << ", explore=" << explore_val << ")\n";
        }
    }

    csv.close();
    std::cout << "\nDBEA shutdown clean. CSV saved as learning_curve.csv" << std::endl;
    return 0;
}