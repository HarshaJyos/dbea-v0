#!/usr/bin/env bash
# run_multiple_seeds.sh - Windows friendly

set -e

EXECUTABLE="./dbea_main.exe"

if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable not found: $EXECUTABLE"
    exit 1
fi

SEEDS=(42 123 456 789 101)

for seed in "${SEEDS[@]}"; do
    LOGFILE="run_seed_${seed}.log"
    echo "Running seed $seed → $LOGFILE"

    "$EXECUTABLE" --seed "$seed" > "$LOGFILE" 2>&1

    # Optional: rename outputs
    if [ -f "gridworld_trajectory.csv" ]; then
        mv "gridworld_trajectory.csv" "gridworld_trajectory_seed_${seed}.csv"
    fi
    if [ -f "emotion_trajectory_full.csv" ]; then
        mv "emotion_trajectory_full.csv" "emotion_trajectory_seed_${seed}.csv"
    fi
done

echo "All done."
echo "Success rates:"
grep "Goals reached" run_seed_*.log