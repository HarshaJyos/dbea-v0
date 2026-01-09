#pragma once
#include <string>
#include "dbea/PatternSignature.h"

namespace dbea {

struct BeliefNode {
    std::string id;
    PatternSignature prototype;

    double confidence;
    double activation;

    BeliefNode(const std::string& id_,
               const PatternSignature& proto)
        : id(id_), prototype(proto),
          confidence(0.5), activation(0.0) {}

    double match_score(const PatternSignature& input) const;
    void reinforce(double amount);
    void decay(double amount);
};

}
