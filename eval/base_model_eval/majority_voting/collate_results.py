import json
import sys


def calculate_hits(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    hits = 0
    total = 0
    
    # Check if data has a 'results' field (nested structure)
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    else:
        results = data
    
    for item in results:
        total += 1
        if item.get("ground_truth") == item.get("predicted_answer"):
            hits += 1
    
    accuracy = hits / total if total > 0 else 0
    
    return {
        "hits": hits,
        "total": total,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_JSON_file = sys.argv[1]
    else:
        results_JSON_file = "/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/evaluation/majority_voting/outputs/majority_voting/mmmu_validation-results/majority_voting_mmmu_validation_0-900_20250815_002603.json"
    
    results = calculate_hits(results_JSON_file)
    print(f"Hits: {results['hits']}/{results['total']}")
    print(f"Accuracy: {results['accuracy']:.2%}")