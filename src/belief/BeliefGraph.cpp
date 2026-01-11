#include "dbea/BeliefGraph.h"
#include <algorithm>
#include <iostream>
#include <memory>

namespace dbea
{

    static int NEXT_BELIEF_ID = 0; // monotonic ID counter

    void BeliefGraph::add_belief(const std::shared_ptr<BeliefNode> &node)
    {
        nodes.push_back(node);
    }

    std::shared_ptr<BeliefNode>
    BeliefGraph::compete(const PatternSignature &input)
    {
        double best_score = -1.0;
        std::shared_ptr<BeliefNode> winner = nullptr;

        for (auto &node : nodes)
        {
            double match = node->match_score(input);
            node->activation = match * node->confidence;

            if (node->activation > best_score)
            {
                best_score = node->activation;
                winner = node;
            }
        }

        return winner;
    }

    std::shared_ptr<BeliefNode>
    BeliefGraph::maybe_create_belief(
        const PatternSignature &input,
        double activation_threshold)
    {
        auto winner = compete(input);

        if (!winner || winner->activation < activation_threshold)
        {
            auto newborn = std::make_shared<BeliefNode>(
                "belief_" + std::to_string(NEXT_BELIEF_ID++),
                input);

            nodes.push_back(newborn);

            std::cout << "[DBEA] New belief created: "
                      << newborn->id << std::endl;

            return newborn;
        }

        return winner;
    }

    void BeliefGraph::prune(double threshold)
    {
        nodes.erase(
            std::remove_if(nodes.begin(), nodes.end(),
                           [threshold](const std::shared_ptr<BeliefNode> &n)
                           {
                               return n->confidence < threshold;
                           }),
            nodes.end());
    }

    void BeliefGraph::merge_beliefs(double merge_threshold)
    {
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            if (nodes[i]->id == "proto-belief")
                continue;

            for (size_t j = i + 1; j < nodes.size();)
            {
                if (nodes[j]->id == "proto-belief")
                {
                    ++j;
                    continue;
                }

                double similarity = nodes[i]->match_score(nodes[j]->prototype);
                if (similarity > merge_threshold && std::abs(nodes[i]->evidence_count - nodes[j]->evidence_count) < 20)
                {
                    
                    int total_evidence = nodes[i]->evidence_count + nodes[j]->evidence_count;

                    // Weighted average prototype
                    for (size_t k = 0; k < nodes[i]->prototype.features.size(); ++k)
                    {
                        nodes[i]->prototype.features[k] =
                            (nodes[i]->prototype.features[k] * nodes[i]->evidence_count +
                             nodes[j]->prototype.features[k] * nodes[j]->evidence_count) /
                            total_evidence;
                    }

                    // Weighted confidence
                    nodes[i]->confidence =
                        (nodes[i]->confidence * nodes[i]->evidence_count +
                         nodes[j]->confidence * nodes[j]->evidence_count) /
                        total_evidence;

                    // Weighted action values (with small smoothing)
                    for (const auto &[action_id, value] : nodes[j]->action_values)
                    {
                        double old_val = nodes[i]->action_values[action_id];
                        nodes[i]->action_values[action_id] =
                            old_val * 0.9 + value * 0.1; // gentle blending
                    }

                    nodes[i]->evidence_count = total_evidence;

                    nodes.erase(nodes.begin() + j);

                    if (config.debug_merging)
                    {
                        std::cout << "[DBEA] Merged: " << nodes[i]->id
                                  << " ← " << " (sim=" << similarity << ", ev=" << total_evidence << ")\n";
                    }
                }
                else
                {
                    ++j;
                }
            }
        }
    }
} // namespace dbea