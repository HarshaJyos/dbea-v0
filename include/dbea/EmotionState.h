#pragma once
#include "dbea/Config.h"  // ← Add this include

struct EmotionState {
    double valence = 0.0;
    double arousal = 0.0;
    double dominance = 0.0;
    double curiosity = 0.5;
    double fear = 0.0;

    EmotionState();
    void update(double reward_valence, double reward_surprise, double avg_error,
                const Config& config);  // ← Added config param
};