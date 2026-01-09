#pragma once

struct EmotionState {
    double valence;    // pleasure ↔ pain
    double arousal;    // calm ↔ excited
    double dominance;  // weak ↔ confident
    double curiosity;
    double fear;

    EmotionState();
    void update(double reward_valence, double reward_surprise);
};
