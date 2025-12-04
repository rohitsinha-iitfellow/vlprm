#!/usr/bin/env python3
"""
Simple GPU memory monitor to detect OOM conditions during parallel evaluation.
Runs alongside the main evaluation processes and logs memory usage.
"""

import subprocess
import time
import sys
import argparse
import os
from datetime import datetime


def get_gpu_memory_info():
    """Get current GPU memory usage for all GPUs."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) >= 5:
                gpu_info.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used': int(parts[2]),
                    'memory_total': int(parts[3]),
                    'memory_percent': (int(parts[2]) / int(parts[3])) * 100,
                    'gpu_util': int(parts[4])
                })
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []


def monitor_memory(interval=30, threshold=95, log_file=None):
    """
    Monitor GPU memory and alert if usage exceeds threshold.
    
    Args:
        interval: Check interval in seconds
        threshold: Memory usage percentage threshold for warnings
        log_file: Optional log file for memory stats
    """
    print(f"Starting GPU memory monitoring (interval={interval}s, threshold={threshold}%)")
    
    if log_file:
        log_handle = open(log_file, 'w')
    else:
        log_handle = sys.stdout
    
    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gpu_info = get_gpu_memory_info()
            
            # Check for high memory usage
            for gpu in gpu_info:
                status = "OK"
                if gpu['memory_percent'] > threshold:
                    status = "WARNING: HIGH MEMORY"
                elif gpu['memory_percent'] > 90:
                    status = "CAUTION"
                    
                log_msg = (f"[{timestamp}] GPU {gpu['index']}: "
                          f"{gpu['memory_used']}MB/{gpu['memory_total']}MB "
                          f"({gpu['memory_percent']:.1f}%) "
                          f"Util: {gpu['gpu_util']}% "
                          f"[{status}]")
                
                print(log_msg, file=log_handle)
                log_handle.flush()
                
                # If memory is critically high, log more details
                if gpu['memory_percent'] > threshold:
                    print(f"⚠️  MEMORY WARNING: GPU {gpu['index']} at {gpu['memory_percent']:.1f}% capacity!", 
                          file=log_handle)
                    log_handle.flush()
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMemory monitoring stopped by user")
    finally:
        if log_file:
            log_handle.close()


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU memory usage")
    parser.add_argument("--interval", type=int, default=30, 
                       help="Check interval in seconds (default: 30)")
    parser.add_argument("--threshold", type=int, default=95,
                       help="Memory warning threshold percentage (default: 95)")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Log file for memory stats (default: stdout)")
    
    args = parser.parse_args()
    
    monitor_memory(args.interval, args.threshold, args.log_file)


if __name__ == "__main__":
    main()