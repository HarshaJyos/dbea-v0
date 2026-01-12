#include "dbea/EmotionState.h"
#include <algorithm>

EmotionState::EmotionState()
    : valence(0.0), arousal(0.0), dominance(0.0), curiosity(0.5), fear(0.0) {}

void EmotionState::update(double reward_valence, double reward_surprise, double avg_error,
                          const Config& config) {
    // Valence update — slightly damped after heavy trauma
    valence += 0.09 * reward_valence;  // was 0.1 → slower climb back from deep negative

    arousal += 0.05 * reward_surprise;

    curiosity = std::clamp(curiosity + config.curiosity_boost * avg_error - config.curiosity_decay,
                           0.0, 1.0);

    // **SLOWER RECOVERY (PTSD-like)**: Base decay -0.003; even slower if fear > 0.3 (-0.0015)
    double fear_decay = (fear > 0.3) ? -0.0015 : -0.003;
    
    // Pharmacological simulation during therapy
    if (config.therapy_mode) {
        fear_decay = -0.05;  // Boosted decay to mimic SSRI/anti-anxiety effect
    }
    
    double fear_delta = 0.3 * (reward_surprise > 0.3 ? 0.15 : 0.25 * avg_error);
    fear = std::clamp(fear + fear_delta + fear_decay, 0.0, 1.0);

    // **Flashback effect**: If fear > 0.3 and negative reward → extra fear/surprise
    if (fear > 0.3 && reward_valence < 0.0) {
        fear = std::clamp(fear + 0.1, 0.0, 1.0);
        arousal = std::clamp(arousal + 0.3, 0.0, 1.0);
    }

    // **Tune dominance drop during trauma**: Slight drop if fear > 0.3
    double dominance_change = 0.12 * (1.0 - avg_error) - 0.04 * avg_error;
    if (fear > 0.3) {
        dominance_change -= 0.05 * avg_error;  // Extra self-doubt during crisis
    }
    dominance += dominance_change;
    dominance = std::clamp(dominance, 0.0, 1.0);

    explore_bias = (curiosity - fear) + 0.3 * valence;
    explore_bias = std::clamp(explore_bias, -1.0, 1.0);

    valence = std::clamp(valence, -1.0, 1.0);
    arousal = std::clamp(arousal, 0.0, 1.0);
}