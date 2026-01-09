#include "dbea/BeliefGraph.h"
#include <algorithm>
#include <iostream>

namespace dbea {

void BeliefGraph::add_belief(
    const std::shared_ptr<BeliefNode>& node) {
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

std::shared_ptr<BeliefNode>
BeliefGraph::maybe_create_belief(
    const PatternSignature& input,
    double activation_threshold) {

    auto winner = compete(input);

    if (!winner || winner->activation < activation_threshold) {
        auto newborn = std::make_shared<BeliefNode>(
            "belief_" + std::to_string(nodes.size()),
            input
        );

        nodes.push_back(newborn);

        std::cout << "[DBEA] New belief created: "
                  << newborn->id << std::endl;

        return newborn;
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

} // namespace dbea
