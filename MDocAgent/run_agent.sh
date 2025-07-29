#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_DIR="/home/haozhuodi/FinM4R/MDocAgent/data/log"
LOG_FILE="${LOG_DIR}/agent-run-doubao-${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

cd /home/haozhuodi/FinM4R/MDocAgent

nohup python scripts/predict.py --config-name ours run-name=test-run-doubao > "$LOG_FILE" 2>&1 &

PID=$!

echo "Agent started with PID: $PID"
echo "Log file: $LOG_FILE"
echo "To monitor the process: tail -f $LOG_FILE"
echo "To stop the process: kill $PID" 