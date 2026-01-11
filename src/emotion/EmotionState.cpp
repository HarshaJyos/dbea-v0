#include "dbea/EmotionState.h"
#include <algorithm>

EmotionState::EmotionState()
    : valence(0.0), arousal(0.0), dominance(0.0), curiosity(0.5), fear(0.0) {}

void EmotionState::update(double reward_valence, double reward_surprise, double avg_error,
                          const Config &config)
{
    valence += 0.1 * reward_valence;
    arousal += 0.05 * reward_surprise;

    curiosity = std::clamp(curiosity + config.curiosity_boost * avg_error - config.curiosity_decay,
                           0.0, 1.0);

    // NEW: Fear from negative surprise / uncertainty
    fear = std::clamp(fear + 0.3 * (reward_surprise > 0.3 ? 0.15 : 0.25 * avg_error) - 0.015, 0.0, 1.0);
    // NEW: Explore bias = curiosity - fear + valence influence
    explore_bias = (curiosity - fear) + 0.3 * valence;
    explore_bias = std::clamp(explore_bias, -1.0, 1.0);

    valence = std::clamp(valence, -1.0, 1.0);
    arousal = std::clamp(arousal, 0.0, 1.0);
}