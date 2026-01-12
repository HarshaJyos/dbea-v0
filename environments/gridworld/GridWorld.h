// environments/gridworld/GridWorld.h
#pragma once
#include "dbea/Environment.h"
#include "dbea/Action.h"
#include "dbea/PatternSignature.h"
#include <array>
#include <utility>

namespace dbea {

class GridWorld : public Environment {
public:
    GridWorld();

    PatternSignature observe() override;
    double step(const Action& action) override;
    bool is_done() override;

    std::pair<int, int> get_position() const { return position; }
    void reset();

private:
    static constexpr int SIZE = 5;
    std::pair<int, int> position{0, 0};
    std::pair<int, int> goal{4, 4};

    // Obstacles (walls) - impassable
    std::array<std::array<bool, SIZE>, SIZE> walls{};

    // Risky tiles (negative reward)
    std::array<std::array<bool, SIZE>, SIZE> risky{};

    // Reward map (for safe tiles)
    double get_tile_reward(int x, int y) const;

    // Helper: try move, return true if successful
    bool try_move(int dx, int dy);
};

}  // namespace dbea