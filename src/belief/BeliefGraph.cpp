#include "dbea/BeliefGraph.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <cmath>

namespace dbea
{
    static int NEXT_BELIEF_ID = 0;

    void BeliefGraph::add_belief(const std::shared_ptr<BeliefNode> &node)
    {
        nodes.push_back(node);
    }

    std::shared_ptr<BeliefNode> BeliefGraph::compete(const PatternSignature &input)
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

    std::shared_ptr<BeliefNode> BeliefGraph::maybe_create_belief(
        const PatternSignature &input,
        double activation_threshold)
    {
        auto winner = compete(input);
        if (!winner || winner->activation < activation_threshold)
        {
            auto newborn = std::make_shared<BeliefNode>(
                "belief_" + std::to_string(NEXT_BELIEF_ID++),
                input);
            newborn->local_lr = 0.1;
            newborn->emotional_affinity = std::vector<double>(5, 0.0);
            nodes.push_back(newborn);
            std::cout << "[DBEA] New belief created: " << newborn->id << std::endl;
            return newborn;
        }
        return winner;
    }

    void BeliefGraph::prune(double threshold)
    {
        nodes.erase(
            std::remove_if(nodes.begin(), nodes.end(),
                           [threshold](const std::shared_ptr<BeliefNode> &n)
                           { return n->confidence < threshold; }),
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
                    for (size_t k = 0; k < nodes[i]->prototype.features.size(); ++k)
                    {
                        nodes[i]->prototype.features[k] =
                            (nodes[i]->prototype.features[k] * nodes[i]->evidence_count +
                             nodes[j]->prototype.features[k] * nodes[j]->evidence_count) /
                            total_evidence;
                    }
                    nodes[i]->confidence =
                        (nodes[i]->confidence * nodes[i]->evidence_count +
                         nodes[j]->confidence * nodes[j]->evidence_count) /
                        total_evidence;
                    for (const auto &[action_id, value] : nodes[j]->action_values)
                    {
                        double old_val = nodes[i]->action_values[action_id];
                        nodes[i]->action_values[action_id] = old_val * 0.9 + value * 0.1;
                    }
                    nodes[i]->evidence_count = total_evidence;
                    nodes.erase(nodes.begin() + j);
                    if (config.debug_merging)
                        std::cout << "[DBEA] Merged: " << nodes[i]->id << " ← (sim=" << similarity << ", ev=" << total_evidence << ")\n";
                }
                else
                    ++j;
            }
        }
    }

    // UPDATED: Now takes emotion reference
    void BeliefGraph::evolve_cycle(const EmotionState &emotion)
    {
        if (nodes.size() < 4)
            return; // Almost no evolution when population tiny

        std::cout << "[NDBE] Starting gentle evolution cycle | Pop: " << nodes.size() << std::endl;

        // Protect proto-belief
        std::shared_ptr<BeliefNode> proto = nullptr;
        for (const auto &n : nodes)
        {
            if (n->id == "proto-belief")
            {
                proto = n;
                break;
            }
        }

        // Compute avg fitness (only positive)
        double total_fitness = 0.0;
        size_t valid_count = 0;
        for (const auto &node : nodes)
        {
            if (node->fitness > 0.0)
            {
                total_fitness += node->fitness;
                valid_count++;
            }
        }
        double avg_fitness = (valid_count > 0) ? total_fitness / valid_count : 1.0;

        // Survivors list — start with everyone
        std::vector<std::shared_ptr<BeliefNode>> survivors = nodes;

        // Only kill clear parasites — very high threshold + evidence protection
        for (auto it = survivors.begin(); it != survivors.end();)
        {
            const auto &b = *it;
            if (b->id == "proto-belief" || b->evidence_count < 8)
            { // Protect young beliefs
                ++it;
                continue;
            }

            double symbiotic_income = 0.0;
            int partner_count = 0;
            for (const auto &[key, count] : co_activations)
            {
                if (key.find(b->id) != std::string::npos && count > 10)
                { // stricter co-activation
                    std::string other_id = (key.substr(0, key.find("_")) == b->id)
                                               ? key.substr(key.find("_") + 1)
                                               : key.substr(0, key.find("_"));
                    for (const auto &other : nodes)
                    {
                        if (other->id == other_id)
                        {
                            symbiotic_income += std::max(0.0, other->fitness);
                            partner_count++;
                            break;
                        }
                    }
                }
            }
            symbiotic_income *= config.symbiotic_uplift / (partner_count + 1e-6);

            double parasite_score = symbiotic_income / (b->fitness + 1e-6);

            // Much higher bar + evidence check
            if (parasite_score > 12.0 && b->fitness < 0.3 * avg_fitness)
            {
                std::cout << "[NDBE] Killed weak parasite: " << b->id
                          << " (score=" << parasite_score << ", fitness=" << b->fitness << ")\n";
                it = survivors.erase(it);
            }
            else
            {
                ++it;
            }
        }

        nodes = std::move(survivors);

        // If population dropped too low, keep at least proto + 2 others
        if (nodes.size() < 4 && proto)
        {
            // ... add some logic if needed, but usually not necessary now
        }

        // Selection — stronger bias toward high fitness
        std::vector<double> selection_probs;
        double prob_sum = 0.0;
        for (const auto &node : nodes)
        {
            double prob = std::pow(std::max(0.01, node->fitness), 2.0) + 0.1; // Square to favor high fitness
            selection_probs.push_back(prob);
            prob_sum += prob;
        }
        for (auto &p : selection_probs)
            p /= prob_sum;

        // Reproduction — top 30% now, 1 child each
        std::vector<std::shared_ptr<BeliefNode>> children;
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::normal_distribution<double> gauss(0.0, 0.8);
        int num_parents = std::max(1, static_cast<int>(nodes.size() * 0.3));

        for (int i = 0; i < num_parents; ++i)
        {
            double roll = uni(rng);
            double cum = 0.0;
            std::shared_ptr<BeliefNode> parent = nullptr;
            for (size_t j = 0; j < nodes.size(); ++j)
            {
                cum += selection_probs[j];
                if (roll <= cum)
                {
                    parent = nodes[j];
                    break;
                }
            }
            if (!parent)
                continue;

            auto child = std::make_shared<BeliefNode>(
                "belief_" + std::to_string(NEXT_BELIEF_ID++),
                parent->prototype);

            // Milder mutation
            for (auto &feat : child->prototype.features)
                feat += gauss(rng) * parent->mutation_rate * 0.4;

            child->action_values = parent->action_values;
            for (auto &[id, val] : child->action_values)
                val += gauss(rng) * parent->mutation_rate * 0.03;

            child->mutation_rate = std::clamp(parent->mutation_rate + gauss(rng) * 0.02, 0.03, 0.35);
            child->local_lr = std::clamp(parent->local_lr + gauss(rng) * 0.008, 0.06, 0.22);

            child->emotional_affinity = parent->emotional_affinity;
            for (auto &aff : child->emotional_affinity)
                aff += gauss(rng) * 0.04;

            child->fitness = parent->fitness * 0.75 + 0.4; // Decent inheritance + boost
            child->confidence = parent->confidence * 0.8;

            children.push_back(child);
        }

        // Horizontal gene transfer (keep but rare)
        for (auto &child : children)
        {
            if (uni(rng) < 0.15)
            { // Reduced prob
                std::uniform_int_distribution<size_t> dist(0, nodes.size() - 1);
                auto target = nodes[dist(rng)];
                if (target == child)
                    continue;

                if (uni(rng) < 0.5 && !target->action_values.empty())
                {
                    auto it = target->action_values.begin();
                    std::advance(it, dist(rng) % target->action_values.size());
                    int rand_id = it->first;
                    std::swap(child->action_values[rand_id], target->action_values[rand_id]);
                }
            }
        }

        // Gentle replacement: kill only 5–15%
        double prune_frac = (avg_fitness < config.crisis_reward_thresh) ? 0.15 : 0.05;
        int num_kill = static_cast<int>(nodes.size() * prune_frac);
        std::sort(nodes.begin(), nodes.end(),
                  [](const auto &a, const auto &b)
                  { return a->fitness < b->fitness; });
        nodes.erase(nodes.begin(), nodes.begin() + num_kill);

        // Guarantee proto survival
        if (proto && std::find(nodes.begin(), nodes.end(), proto) == nodes.end())
        {
            nodes.push_back(proto);
        }

        // Add children
        for (auto &child : children)
            add_belief(child);

        // Arousal boost (milder)
        if (emotion.arousal > 0.7)
        {
            for (auto &node : nodes)
                node->mutation_rate = std::min(0.45, node->mutation_rate * 1.1);
        }

        std::cout << "[NDBE] Cycle complete | Killed: " << num_kill << " | Born: " << children.size()
                  << " | New pop: " << nodes.size() << " | Avg fitness: " << avg_fitness << std::endl;
    }
} // namespace dbea