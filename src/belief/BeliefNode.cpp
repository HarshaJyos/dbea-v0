#include "dbea/BeliefNode.h"

BeliefNode::BeliefNode(const std::string& id_) : id(id_), confidence(0.5), activation(0.0) {}

void BeliefNode::update_activation(double delta) {
    activation += delta;
}
