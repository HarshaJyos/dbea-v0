#pragma once
#include "dbea/BeliefNode.h"
#include <vector>
#include <memory>

class BeliefGraph {
public:
    std::vector<std::shared_ptr<BeliefNode>> nodes;

    BeliefGraph();
    void add_belief(const std::shared_ptr<BeliefNode>& node);
    void prune_beliefs();
    void activate_beliefs();
};
