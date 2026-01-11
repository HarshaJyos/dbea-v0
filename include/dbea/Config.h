#pragma once

struct Config
{
    int max_beliefs = 100;
    double exploration_rate = 0.8;
    double learning_rate = 0.1;
    double belief_learning_rate = 0.05;
    double belief_decay_rate = 0.01;
    double merge_threshold = 0.95; // Adjustable
    bool debug_merging = true;     // NEW: toggle merge logs
    double curiosity_boost = 0.4;
    double curiosity_decay = 0.01;
    double curiosity_threshold = 0.4;      // when to lower creation threshold
    double curiosity_threshold_drop = 0.2; // strength of drop
    double max_explore_streak = 4;
    double streak_punish_prob = 0.3;
    double streak_punish_amount = -0.2;
};