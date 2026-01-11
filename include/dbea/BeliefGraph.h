#pragma once
#include <vector>
#include <memory>
#include "dbea/BeliefNode.h"
#include "dbea/PatternSignature.h"
#include "dbea/Config.h"

namespace dbea {

class BeliefGraph {
public:
    BeliefGraph(const Config& cfg) : config(cfg) {}  // ← Inline definition — correct

    std::vector<std::shared_ptr<BeliefNode>> nodes;

    void add_belief(const std::shared_ptr<BeliefNode>& node);
    std::shared_ptr<BeliefNode> compete(const PatternSignature& input);
    std::shared_ptr<BeliefNode> maybe_create_belief(const PatternSignature& input,
                                                    double activation_threshold);
    void prune(double threshold = 0.25);
    void merge_beliefs(double merge_threshold = 0.95);

private:
    const Config& config;
};

} // namespace dbea