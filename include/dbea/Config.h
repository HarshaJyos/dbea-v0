#pragma once
struct Config
{
    int max_beliefs = 100;
    double exploration_rate = 0.92;
    double learning_rate = 0.1;
    double belief_learning_rate = 0.05;
    double belief_decay_rate = 0.01;
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
    double min_exploration = 0.35;
    double explore_bias_scale = 0.45;
    double gamma = 0.98;
    int min_beliefs_before_prune = 12;
    // NEW: Evolutionary parameters
    int evo_cycle_freq = 50;  // Evolve every N steps
    double crisis_reward_thresh = -0.5;  // Trigger mass extinction if avg reward < this
    double niche_bonus_scale = 0.3;  // ξ
    double symbiotic_uplift = 0.15;  // ζ
    double niche_radius = 0.2;  // Distance threshold for niche density
    double co_activation_thresh = 0.3;  // For symbiosis detection
    double symbiosis_prob = 0.3;  // Prob of horizontal transfer
    // Parasite prevention
    double parasite_tau = 4.5;  // τ threshold
    double parasite_phi = 0.1;  // ϕ min intrinsic fitness (relative to avg)
};
