#!/bin/bash

# Monitor GPU usage during Llama model inference
# This script will show GPU usage statistics in a continuous loop

echo "Llama 3 GPU Monitor"
echo "==================="
echo "This script will continuously monitor GPU usage."
echo "Press Ctrl+C to stop monitoring."
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi command not found. NVIDIA drivers may not be installed correctly."
    exit 1
fi

# Function to display GPU info
show_gpu_info() {
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader | \
    awk -F", " '{printf "GPU %d: %s\n  Memory: %s / %s\n  Utilization: %s\n", NR, $1, $3, $2, $4}'
    echo ""
}

# Function to display processes using GPU
show_gpu_processes() {
    echo "Processes using GPU:"
    nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader | \
    while IFS="," read -r pid memory uuid; do
        name=$(ps -p $pid -o comm= 2>/dev/null || echo "Unknown")
        echo "  PID $pid ($name): $memory"
    done
    
    if [ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -eq 0 ]; then
        echo "  No GPU processes found"
    fi
    echo ""
}

# Main loop
clear
while true; do
    clear
    date
    echo "==================="
    
    show_gpu_info
    show_gpu_processes
    
    echo "Press Ctrl+C to exit."
    echo "Refreshing in 2 seconds..."
    
    sleep 2
done 