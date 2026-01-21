#pragma once
struct Config
{
    int max_beliefs = 300;
    double exploration_rate = 0.92;
    double learning_rate = 0.1;
    double belief_learning_rate = 0.04;
    double belief_decay_rate = 0.015;
    double merge_threshold = 0.92;
    bool debug_merging = true;
    double curiosity_boost = 0.65;
    double curiosity_decay = 0.008;
    double curiosity_threshold = 0.25;
    double curiosity_threshold_drop = 0.2;
    double max_explore_streak = 5;
    double streak_punish_prob = 0.25;
    double streak_punish_amount = -0.15;
    bool therapy_mode = false;
    double epsilon_decay = 0.9995;
    double min_exploration = 0.42;
    double explore_bias_scale = 0.45;
    double gamma = 0.985;
    int min_beliefs_before_prune = 30;
    // NEW: Evolutionary parameters
    // Evolutionary parameters — tuned for stability
    int evo_cycle_freq = 800;           // Much rarer — was 50
    double crisis_reward_thresh = -1.5; // Harder to trigger crisis
    double niche_bonus_scale = 0.20;
    double symbiotic_uplift = 0.12;
    double niche_radius = 0.25;
    double co_activation_thresh = 0.35;
    double symbiosis_prob = 0.12; // Less frequent transfer

    // Parasite prevention — much less aggressive
    double parasite_tau = 10.0; // Very high threshold
    double parasite_phi = 0.08; // Lower floor
};
