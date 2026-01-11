#pragma once

struct Config
{
    int max_beliefs;
    double exploration_rate;
    double learning_rate;
    double belief_learning_rate = 0.05;
    double belief_decay_rate = 0.01;
    double merge_threshold = 0.95;  // NEW: Threshold for belief merging
};