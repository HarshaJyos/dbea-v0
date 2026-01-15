#include "dbea/EmotionState.h"
#include <algorithm>
#include <cmath>  // for std::isfinite

EmotionState::EmotionState()
    : valence(0.0), arousal(0.0), dominance(0.5), curiosity(0.5), fear(0.0), explore_bias(0.0) {}

void EmotionState::update(double reward_valence, double reward_surprise, double avg_error,
                          const Config& config)
{
    // Valence update
    valence += 0.09 * reward_valence;
    valence = std::clamp(valence, -1.0, 1.0);

    // Arousal update
    arousal += 0.05 * reward_surprise;
    arousal = std::clamp(arousal, 0.0, 1.0);

    // Curiosity update — protect against NaN avg_error
    double safe_error = std::isfinite(avg_error) ? avg_error : 0.0;
    curiosity += config.curiosity_boost * safe_error - config.curiosity_decay;
    curiosity = std::clamp(curiosity, 0.0, 1.0);

    // Fear update
    double fear_decay = (fear > 0.3) ? -0.0015 : -0.003;
    if (config.therapy_mode) {
        fear_decay = -0.05;
    }
    double fear_delta = 0.3 * (reward_surprise > 0.3 ? 0.15 : 0.25 * safe_error);
    fear += fear_delta + fear_decay;
    fear = std::clamp(fear, 0.0, 1.0);

    // Flashback effect
    if (fear > 0.3 && reward_valence < 0.0) {
        fear = std::clamp(fear + 0.1, 0.0, 1.0);
        arousal = std::clamp(arousal + 0.3, 0.0, 1.0);
    }

    // Dominance update
    double dominance_change = 0.12 * (1.0 - safe_error) - 0.04 * safe_error;
    if (fear > 0.3) {
        dominance_change -= 0.05 * safe_error;
    }
    dominance += dominance_change;
    dominance = std::clamp(dominance, 0.0, 1.0);

    // Explore bias
    explore_bias = (curiosity - fear) + 0.3 * valence;
    explore_bias = std::clamp(explore_bias, -1.0, 1.0);

    // FINAL SAFETY NET: Force sane values if anything goes NaN
    if (!std::isfinite(valence)) valence = 0.0;
    if (!std::isfinite(arousal)) arousal = 0.0;
    if (!std::isfinite(curiosity)) curiosity = 0.5;
    if (!std::isfinite(fear)) fear = 0.0;
    if (!std::isfinite(dominance)) dominance = 0.5;
    if (!std::isfinite(explore_bias)) explore_bias = 0.0;
}