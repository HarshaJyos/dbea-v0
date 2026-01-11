#include "dbea/BeliefGraph.h"
#include <algorithm>
#include <iostream>
#include <memory>

namespace dbea {

static int NEXT_BELIEF_ID = 0;  // monotonic ID counter

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

    return winner;
}

std::shared_ptr<BeliefNode>
BeliefGraph::maybe_create_belief(
    const PatternSignature& input,
    double activation_threshold
) {
    auto winner = compete(input);

    if (!winner || winner->activation < activation_threshold) {
        auto newborn = std::make_shared<BeliefNode>(
            "belief_" + std::to_string(NEXT_BELIEF_ID++),
            input
        );

        nodes.push_back(newborn);

        std::cout << "[DBEA] New belief created: "
                  << newborn->id << std::endl;

        return newborn;
    }

    return winner;
}

void BeliefGraph::prune(double threshold) { // NO default value here!
    nodes.erase(
        std::remove_if(nodes.begin(), nodes.end(),
            [threshold](const std::shared_ptr<BeliefNode>& n) {
                return n->confidence < threshold;
            }),
        nodes.end()
    );
}

void BeliefGraph::merge_beliefs(double merge_threshold) {
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = i + 1; j < nodes.size(); ) {
            double similarity = nodes[i]->match_score(nodes[j]->prototype);
            if (similarity > merge_threshold) {
                // Merge j into i (weighted by evidence_count)
                int total_evidence = nodes[i]->evidence_count + nodes[j]->evidence_count;

                // Average prototype
                for (size_t k = 0; k < nodes[i]->prototype.features.size(); ++k) {
                    nodes[i]->prototype.features[k] = 
                        (nodes[i]->prototype.features[k] * nodes[i]->evidence_count +
                         nodes[j]->prototype.features[k] * nodes[j]->evidence_count) / total_evidence;
                }

                // Weighted confidence
                nodes[i]->confidence = 
                    (nodes[i]->confidence * nodes[i]->evidence_count +
                     nodes[j]->confidence * nodes[j]->evidence_count) / total_evidence;

                // Average action values
                for (const auto& [action_id, value] : nodes[j]->action_values) {
                    double old_val = nodes[i]->action_values[action_id];
                    nodes[i]->action_values[action_id] = 
                        (old_val * nodes[i]->evidence_count + value * nodes[j]->evidence_count) / total_evidence;
                }

                // Update evidence count
                nodes[i]->evidence_count = total_evidence;

                // Remove j
                nodes.erase(nodes.begin() + j);
                std::cout << "[DBEA] Merged beliefs: " << nodes[i]->id << " absorbed another belief" << std::endl;
            } else {
                ++j;
            }
        }
    }
}

} // namespace dbea
