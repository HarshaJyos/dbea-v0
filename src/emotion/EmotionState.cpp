#include "dbea/EmotionState.h"
#include <algorithm>

EmotionState::EmotionState() 
    : valence(0.0), arousal(0.0), dominance(0.0), curiosity(0.5), fear(0.0) {}

void EmotionState::update(double reward_valence, double reward_surprise, double avg_error,
                          const Config& config) {
    valence += 0.1 * reward_valence;
    arousal += 0.05 * reward_surprise;

    curiosity = std::clamp(curiosity + config.curiosity_boost * avg_error - config.curiosity_decay,
                           0.0, 1.0);

    valence = std::clamp(valence, -1.0, 1.0);
    arousal = std::clamp(arousal, 0.0, 1.0);
}