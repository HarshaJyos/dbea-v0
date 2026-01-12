// environments/gridworld/GridWorld.cpp
#include "GridWorld.h"
#include <cmath>

namespace dbea {

GridWorld::GridWorld() {
    reset();

    // Walls (obstacles)
    walls[1][2] = true;
    walls[2][2] = true;
    walls[3][1] = true;

    // Risky tiles
    risky[2][3] = true;
    risky[3][3] = true;
    risky[4][2] = true;
}

void GridWorld::reset() {
    position = {0, 0};
}

PatternSignature GridWorld::observe() {
    double norm_x = static_cast<double>(position.first) / (SIZE - 1);
    double norm_y = static_cast<double>(position.second) / (SIZE - 1);
    return PatternSignature{{norm_x, norm_y}};
}

double GridWorld::get_tile_reward(int x, int y) const {
    if (x == goal.first && y == goal.second) return 1.0;
    if (risky[x][y]) return -0.1;
    return 0.08;  // ← was 0.05 — stronger movement incentive
}

bool GridWorld::try_move(int dx, int dy) {
    int nx = position.first + dx;
    int ny = position.second + dy;

    if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE) return false;
    if (walls[nx][ny]) return false;

    position = {nx, ny};
    return true;
}

double GridWorld::step(const Action& action) {
    if (is_done()) return 0.0;
    int dx = 0, dy = 0;
    if (action.name == "up")    dy = -1;
    if (action.name == "down")  dy =  1;
    if (action.name == "left")  dx = -1;
    if (action.name == "right") dx =  1;

    bool moved = try_move(dx, dy);
    double reward = moved ? get_tile_reward(position.first, position.second)
                          : -0.015;  // ← was -0.02 — milder wall-bump penalty
    return reward;
}

bool GridWorld::is_done() {
    return position.first == goal.first && position.second == goal.second;
}

}  // namespace dbea