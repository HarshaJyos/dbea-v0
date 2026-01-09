#pragma once

#include <string>
#include <cstdint>

namespace dbea {

/**
 * @brief Represents an executable action available to the agent.
 *
 * Actions are intentionally simple in v0.
 * Intelligence lives in belief selection, not in actions themselves.
 */
struct Action {
    uint32_t id;              // Stable identifier
    std::string name;         // Human-readable label
    double energy_cost;       // Metabolic / effort cost
    double risk;              // Expected danger / uncertainty
    double novelty;           // Encourages exploration (childhood bias)

    Action()
        : id(0),
          energy_cost(0.0),
          risk(0.0),
          novelty(0.0) {}

    Action(uint32_t id_,
           const std::string& name_,
           double energy_cost_ = 0.0,
           double risk_ = 0.0,
           double novelty_ = 0.0)
        : id(id_),
          name(name_),
          energy_cost(energy_cost_),
          risk(risk_),
          novelty(novelty_) {}
};

} // namespace dbea
