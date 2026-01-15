#pragma once
#include <vector>
#include <memory>
#include <unordered_map>
#include "dbea/BeliefNode.h"
#include "dbea/PatternSignature.h"
#include "dbea/Config.h"
#include "dbea/EmotionState.h"  // NEW: needed for evolve_cycle param

namespace dbea {
class BeliefGraph {
public:
    BeliefGraph(const Config& cfg) : config(cfg) {}
    
    std::vector<std::shared_ptr<BeliefNode>> nodes;
    
    // NEW: Made public so Agent can access it directly for symbiosis tracking
    std::unordered_map<std::string, int> co_activations;  // "id1_id2" → count

    void add_belief(const std::shared_ptr<BeliefNode>& node);
    std::shared_ptr<BeliefNode> compete(const PatternSignature& input);
    std::shared_ptr<BeliefNode> maybe_create_belief(const PatternSignature& input,
                                                    double activation_threshold);
    void prune(double threshold = 0.25);
    void merge_beliefs(double merge_threshold = 0.95);

    // NEW: Now takes emotion reference for arousal-based scaling
    void evolve_cycle(const EmotionState& emotion);

private:
    const Config& config;
};
} // namespace dbea