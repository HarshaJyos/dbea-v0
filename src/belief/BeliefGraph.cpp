#include "dbea/BeliefGraph.h"
#include <algorithm>

namespace dbea {

void BeliefGraph::add_belief(const std::shared_ptr<BeliefNode>& node) {
    nodes.push_back(node);
}

std::shared_ptr<BeliefNode>
BeliefGraph::compete(const PatternSignature& input) {

    double best_score = -1.0;
    std::shared_ptr<BeliefNode> winner = nullptr;

    for (auto& node : nodes) {
        double match = node->match_score(input);
        node->activation = match * node->confidence;

        if (node->activation > best_score) {
            best_score = node->activation;
            winner = node;
        }
    }

    for (auto& node : nodes) {
        if (node == winner) {
            node->reinforce(0.05);
        } else {
            node->decay(0.02);
        }
    }

    return winner;
}

void BeliefGraph::prune(double threshold) {
    nodes.erase(
        std::remove_if(nodes.begin(), nodes.end(),
            [threshold](const std::shared_ptr<BeliefNode>& n) {
                return n->confidence < threshold;
            }),
        nodes.end());
}

}
