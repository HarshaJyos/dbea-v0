#pragma once
#include <string>
#include <vector>

struct BeliefNode {
    std::string id;
    double confidence;
    double activation;
    
    BeliefNode(const std::string& id_);
    void update_activation(double delta);
};
