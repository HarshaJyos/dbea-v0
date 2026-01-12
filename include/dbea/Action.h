// include/dbea/Action.h
#pragma once
#include <string>
#include <cstdint>

namespace dbea {

struct Action {
    uint32_t id;
    std::string name;
    double energy_cost = 0.05;  // small cost for each move
    double risk = 0.0;
    double novelty = 0.0;

    Action() = default;
    Action(uint32_t id_, const std::string& name_,
           double energy_cost_ = 0.05, double risk_ = 0.0, double novelty_ = 0.0)
        : id(id_), name(name_), energy_cost(energy_cost_), risk(risk_), novelty(novelty_) {}
};

}  // namespace dbea