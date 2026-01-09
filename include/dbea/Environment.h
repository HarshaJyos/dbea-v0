#pragma once
#include "dbea/PatternSignature.h"
#include "dbea/Action.h"
#include <vector>

namespace dbea {

class Environment {
public:
    virtual ~Environment() = default;
    virtual PatternSignature observe() = 0;
    virtual double step(const Action& action) = 0;
    virtual bool is_done() = 0;
};

}

