#pragma once
#include <vector>
#include <memory>
#include "dbea/BeliefNode.h"
#include "dbea/PatternSignature.h"

namespace dbea {

class BeliefGraph {
public:
    std::vector<std::shared_ptr<BeliefNode>> nodes;

    void add_belief(const std::shared_ptr<BeliefNode>& node);
    std::shared_ptr<BeliefNode> compete(const PatternSignature& input);
    void prune(double threshold = 0.1);
};

}
