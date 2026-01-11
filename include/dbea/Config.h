#pragma once

struct Config
{
    int max_beliefs = 100;
    double exploration_rate = 0.8;
    double learning_rate = 0.1;
    double belief_learning_rate = 0.05;
    double belief_decay_rate = 0.01;
    double merge_threshold = 0.95;          // Adjustable
    bool debug_merging = true;              // NEW: toggle merge logs
};