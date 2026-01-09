#include "dbea/EmotionState.h"
#include <algorithm>

EmotionState::EmotionState() 
    : valence(0.0), arousal(0.0), dominance(0.0), curiosity(0.5), fear(0.0) {}

void EmotionState::update(double reward_valence, double reward_surprise) {
    valence += 0.1 * reward_valence;
    arousal += 0.05 * reward_surprise;
    // clamp between -1.0 and 1.0
    valence = std::max(-1.0, std::min(1.0, valence));
    arousal = std::max(0.0, std::min(1.0, arousal));
}
