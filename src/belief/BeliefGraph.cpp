#include "dbea/BeliefGraph.h"

BeliefGraph::BeliefGraph() {}

void BeliefGraph::add_belief(const std::shared_ptr<BeliefNode>& node) {
    nodes.push_back(node);
}

void BeliefGraph::prune_beliefs() {
    // TODO: prune low confidence beliefs
}

void BeliefGraph::activate_beliefs() {
    for (auto& node : nodes) {
        // TODO: update node->activation based on input patterns
    }
}
