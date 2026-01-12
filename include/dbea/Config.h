#pragma once

struct Config
{
    int max_beliefs = 100;
    double exploration_rate     = 0.92;          // high initial randomness
    double learning_rate        = 0.1;
    double belief_learning_rate = 0.05;
    double belief_decay_rate    = 0.01;
    double merge_threshold      = 0.95;
    bool debug_merging          = true;

    double curiosity_boost      = 0.65;          // stronger intrinsic motivation
    double curiosity_decay      = 0.008;         // slower decay → longer curiosity
    double curiosity_threshold  = 0.25;          // trigger new beliefs easier
    double curiosity_threshold_drop = 0.2;

    double max_explore_streak   = 4;
    double streak_punish_prob   = 0.3;
    double streak_punish_amount = -0.2;

    bool therapy_mode           = false;

    // Epsilon-greedy parameters
    double epsilon_decay        = 0.998;         // slow decay
    double min_exploration      = 0.25;          // floor at 25% random

    // Emotional scaling
    double explore_bias_scale   = 0.45;          // stronger emotional push

    // Discount factor for TD learning
    double gamma                = 0.95;          // classic gridworld value
};