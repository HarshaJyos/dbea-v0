#pragma once
#include "../include/dbea/Environment.h"

class GridWorld : public Environment {
public:
    PatternSignature observe() override;
    double step(const Action& action) override;
    bool is_done() override;
};
