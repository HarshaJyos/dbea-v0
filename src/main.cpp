#include "dbea/Agent.h"
#include "dbea/Config.h"
#include <iostream>

using namespace dbea;

int main() {
    Config cfg;
    cfg.max_beliefs = 100;
    cfg.exploration_rate = 0.8;
    cfg.learning_rate = 0.1;

    Agent agent(cfg);

    std::cout << "DBEA v0 bootstrap running." << std::endl;

    // Dummy perception loop
    for (int step = 0; step < 5; ++step) {
        PatternSignature perception({0.1 * step, 0.2});
        agent.perceive(perception);

        Action action = agent.decide();
        agent.receive_reward(0.1, 0.05);
        agent.learn();

        std::cout << "Step " << step
                  << " | Action: " << action.name << std::endl;
    }

    std::cout << "DBEA shutdown clean." << std::endl;
    return 0;
}
