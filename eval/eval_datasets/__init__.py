from pathlib import Path

DATASETS_ROOT = Path(__file__).parent
PUZZLE_TEST_ROOT = DATASETS_ROOT / "LLM-PuzzleTest"
PUZZLE_VQA_DATA = PUZZLE_TEST_ROOT / "PuzzleVQA" / "data"
ALGO_PUZZLE_VQA_DATA = PUZZLE_TEST_ROOT / "AlgoPuzzleVQA" / "data"

