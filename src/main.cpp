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

    // Random generator for perception noise
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> noise(-0.05, 0.05);

    // Open CSV for recording proto-belief action values
    std::ofstream csv("learning_curve.csv");
    csv << "Episode,Step,Action_noop,Action_explore\n";
    csv << std::fixed << std::setprecision(6);

    for (int ep = 0; ep < episodes; ++ep)
    {
        std::cout << "\n=== Episode " << ep << " ===" << std::endl;

        for (int step = 0; step < steps_per_episode; ++step)
        {
            // Generate perception with small variation
            PatternSignature perception({0.1 * step + noise(rng),
                                         0.2 + noise(rng)});

            // Agent perceives
            agent.perceive(perception);

            // Decide action
            Action action = agent.decide();

            // Dynamic reward: explore gets higher reward on even steps
            double reward_valence = 0.05 + 0.15 * ((action.name == "explore") && (step % 2 == 0));
            double reward_surprise = 0.05;

            // Provide reward
            agent.receive_reward(reward_valence, reward_surprise);

            // Update beliefs
            agent.learn();

            // Log to console
            std::cout << "Step " << step
                      << " | Action: " << action.name
                      << " | Reward: " << reward_valence << std::endl;

            // Log proto-belief action values to CSV using getter
            auto [noop_val, explore_val] = agent.get_proto_action_values();
            std::string top_action = (explore_val >= noop_val) ? "explore" : "noop";

            std::cout << "Proto-belief"
                      << " | Top Action: " << top_action
                      << " | Values: (noop=" << noop_val
                      << ", explore=" << explore_val << ")\n";
        }

        // Episode summary: top action per belief
        std::cout << "\n[Episode " << ep << " Summary]\n";
        // Loop through beliefs using local copies of predicted action values
        for (size_t i = 0; i < agent.get_proto_action_values().first + 1; ++i)
        {
            // Note: we only have proto-belief values accessible for now;
            // for full belief summaries, Agent could provide a similar getter for all beliefs.
            auto [noop_val, explore_val] = agent.get_proto_action_values();
            std::string top_action = (explore_val >= noop_val) ? "explore" : "noop";
            std::cout << "Belief " << i
                      << " | Top Action: " << top_action
                      << " | Values: (noop=" << noop_val
                      << ", explore=" << explore_val << ")\n";
        }
    }

    csv.close();
    std::cout << "\nDBEA shutdown clean. CSV saved as learning_curve.csv" << std::endl;
    return 0;
}
