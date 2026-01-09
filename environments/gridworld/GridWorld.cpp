#include "GridWorld.h"
#include "dbea/Environment.h"

PatternSignature GridWorld::observe() {
    // TODO: return state features
    return PatternSignature();
}

double GridWorld::step(const Action& action) {
    // TODO: update environment, return reward
    return 0.0;
}

bool GridWorld::is_done() {
    // TODO: terminal condition
    return false;
}
