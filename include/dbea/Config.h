#pragma once
struct Config
{
    int max_beliefs = 100;
    double exploration_rate = 0.92;
    double learning_rate = 0.1;
    double belief_learning_rate = 0.05;
    double belief_decay_rate = 0.01;

    double merge_threshold = 0.92;              // ← lowered from 0.95
    bool debug_merging = true;

    double curiosity_boost = 0.65;
    double curiosity_decay = 0.008;
    double curiosity_threshold = 0.25;
    double curiosity_threshold_drop = 0.2;

    double max_explore_streak = 5;              // ← slightly increased
    double streak_punish_prob = 0.25;           // ← reduced punish probability
    double streak_punish_amount = -0.15;        // ← milder punishment

    bool therapy_mode = false;

    // Epsilon-greedy parameters — slower decay, higher floor
    double epsilon_decay = 0.9995;              // ← was 0.998
    double min_exploration = 0.35;              // ← was 0.25 — stronger persistent exploration

    double explore_bias_scale = 0.45;

    // Discount factor — increased for better long-term planning
    double gamma = 0.98;                        // ← was 0.95

    // NEW: Belief management tuning
    int min_beliefs_before_prune = 12;          // don't prune aggressively when few beliefs exist
};