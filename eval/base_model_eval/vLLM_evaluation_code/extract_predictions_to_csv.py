#!/usr/bin/env python3
"""
Script to extract 'raw_full_prediction' values from a JSON file and save to CSV.
"""

import json
import csv
import argparse
from pathlib import Path


def extract_predictions(json_file_path, csv_file_path):
    """
    Load JSON file and extract raw_full_prediction values to CSV.
    
    Args:
        json_file_path: Path to the input JSON file
        csv_file_path: Path to the output CSV file
    """
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both single object and array of objects
    if isinstance(data, dict):
        data = [data]
    
    # Extract raw_full_prediction values
    predictions = []
    for i, item in enumerate(data):
        prediction = item.get('raw_full_prediction', '')
        # Also include index if available for reference
        index = item.get('index', i)
        predictions.append({
            'index': index,
            'raw_full_prediction': prediction
        })
    
    # Write to CSV file (vertically, one row per prediction)
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['index', 'raw_full_prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for pred in predictions:
            writer.writerow(pred)
    
    print(f"Successfully extracted {len(predictions)} predictions")
    print(f"Saved to: {csv_file_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract raw_full_prediction values from JSON to CSV'
    )
    parser.add_argument(
        'input_json',
        type=str,
        help='Path to the input JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to the output CSV file (default: input_name_predictions.csv)'
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        input_path = Path(args.input_json)
        output_path = input_path.parent / f"{input_path.stem}_predictions.csv"
    else:
        output_path = Path(args.output)
    
    # Check if input file exists
    if not Path(args.input_json).exists():
        print(f"Error: Input file '{args.input_json}' not found")
        return 1
    
    # Extract predictions
    try:
        extract_predictions(args.input_json, output_path)
        return 0
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1


if __name__ == "__main__":
    exit(main())