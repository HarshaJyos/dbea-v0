#pragma once
struct Config
{
    int max_beliefs = 100;
    double exploration_rate = 0.92;
    double learning_rate = 0.1;
    double belief_learning_rate = 0.05;
    double belief_decay_rate = 0.01;
    double merge_threshold = 0.965;
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
    int min_beliefs_before_prune = 12;
    // NEW: Evolutionary parameters
    // Evolutionary parameters — tuned for stability
    int evo_cycle_freq = 350;           // Much rarer — was 50
    double crisis_reward_thresh = -2.0; // Harder to trigger crisis
    double niche_bonus_scale = 0;
    double symbiotic_uplift = 0;
    double niche_radius = 0.25;
    double co_activation_thresh = 0.35;
    double symbiosis_prob = 0.18; // Less frequent transfer

    // Parasite prevention — much less aggressive
    double parasite_tau = 15.0; // Very high threshold
    double parasite_phi = 0.08; // Lower floor
};
