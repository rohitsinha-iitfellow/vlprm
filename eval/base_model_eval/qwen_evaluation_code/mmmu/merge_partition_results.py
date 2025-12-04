#!/usr/bin/env python3
"""
Merge partition results from parallel MMMU inference runs.
Combines JSONL files from multiple partitions into a single sorted file.
"""

import os
import json
import glob
import argparse
from typing import List, Dict, Any


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of dictionaries."""
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return results


def merge_partition_results(args):
    """Merge partition result files into a single file."""
    
    # Find all partition files
    pattern = os.path.join(args.output_dir, f"results-p*-*.jsonl")
    partition_files = sorted(glob.glob(pattern))
    
    print(f"Looking for partition files matching: {pattern}")
    print(f"Found {len(partition_files)} partition files")
    
    if not partition_files:
        print(f"ERROR: No partition files found matching pattern: {pattern}")
        return 1
    
    if len(partition_files) != args.num_partitions:
        print(f"WARNING: Expected {args.num_partitions} partitions but found {len(partition_files)}")
        print("Continuing with available partitions...")
    
    # Display found partition files
    print("\nPartition files to merge:")
    for i, file in enumerate(partition_files):
        size = os.path.getsize(file)
        print(f"  {i+1}. {os.path.basename(file)} ({size:,} bytes)")
    
    # Load and merge all partitions
    all_results = []
    partition_stats = []
    
    for file in partition_files:
        try:
            data = load_jsonl(file)
            if data:
                all_results.extend(data)
                partition_stats.append({
                    'file': os.path.basename(file),
                    'samples': len(data)
                })
                print(f"  Loaded {len(data)} samples from {os.path.basename(file)}")
            else:
                print(f"  WARNING: No data in {os.path.basename(file)}")
        except Exception as e:
            print(f"  ERROR loading {file}: {e}")
            continue
    
    if not all_results:
        print("ERROR: No data could be loaded from partition files")
        return 1
    
    # Sort results by index to maintain original order
    try:
        all_results.sort(key=lambda x: int(x.get('index', 0)))
        print(f"\nSorted {len(all_results)} total results by index")
    except Exception as e:
        print(f"Warning: Could not sort by index: {e}")
        print("Results will be in partition order")
    
    # Write merged file
    merged_filename = "results_merged.jsonl"
    if args.timestamp:
        merged_filename = f"results_merged_{args.timestamp}.jsonl"
    
    merged_path = os.path.join(args.output_dir, merged_filename)
    
    try:
        with open(merged_path, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\nMerged file written: {merged_path}")
        print(f"Total samples in merged file: {len(all_results)}")
    except Exception as e:
        print(f"ERROR writing merged file: {e}")
        return 1
    
    # Write statistics file
    stats_path = os.path.join(args.output_dir, "merge_statistics.json")
    stats = {
        'dataset': args.dataset,
        'timestamp': args.timestamp,
        'num_partitions_expected': args.num_partitions,
        'num_partitions_found': len(partition_files),
        'total_samples': len(all_results),
        'partition_details': partition_stats,
        'merged_file': merged_filename
    }
    
    try:
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Statistics written to: {stats_path}")
    except Exception as e:
        print(f"Warning: Could not write statistics file: {e}")
    
    # Display summary
    print("\n" + "="*50)
    print("Merge Summary:")
    print(f"  Partitions merged: {len(partition_files)}")
    print(f"  Total samples: {len(all_results)}")
    for stat in partition_stats:
        print(f"  - {stat['file']}: {stat['samples']} samples")
    print("="*50)
    
    # Verify data integrity
    indices = [r.get('index') for r in all_results if 'index' in r]
    if indices:
        unique_indices = set(indices)
        if len(unique_indices) != len(indices):
            print(f"\nWARNING: Found duplicate index values!")
            print(f"Total indices: {len(indices)}, Unique indices: {len(unique_indices)}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Merge partition results from parallel MMMU inference"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing partition result files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (for documentation)"
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=8,
        help="Expected number of partitions (default: 8)"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp for output filename"
    )
    
    args = parser.parse_args()
    
    # Validate output directory exists
    if not os.path.exists(args.output_dir):
        print(f"ERROR: Output directory does not exist: {args.output_dir}")
        return 1
    
    # Run merge
    return merge_partition_results(args)


if __name__ == "__main__":
    exit(main())