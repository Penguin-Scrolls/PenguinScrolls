import os

ERROR_PREFIX = "@|@|@INTERNAL_ERROR@|@|@"  # Unique string prefix used to identify internal error messages

DEFAULT_DATASET_NAME = "penguin_scrolls"  # Default name of the dataset if not specified in environment
DATASET_NAME = os.environ.get("PENGUIN_SCROLLS", DEFAULT_DATASET_NAME)  # Actual dataset name from env or default
DEFAULT_SPLIT = "test"  # Default dataset split to use (e.g., train/test/validation)
PENGUIN_EVAL_MODEL = os.environ.get("PENGUIN_EVAL_MODEL", "gpt-4o")  # Model used for evaluation, from env or default

INPUT_COL = "prompt"  # Column name for the input data
KEY_COL = "input_md5"  # Column name for MD5 hash of input (used as unique identifier)
QUESTION_COL = "question"  # Column name for the question/prompt text
ANSWER_COL = "answer"  # Column name for the expected answer/output
RESPONSE_COL = "response"  # Column name for the actual model response
EVAL_RESULT_COL = "result"  # Column name for evaluation result details
SCORE_COL = "score"  # Column name for numerical evaluation score
ERROR_COL = "error"  # Column name for error messages if any
TRUNCATED_COL = "truncated"  # Column name indicating if response was truncated
IS_SUBJECTIVE_COL = "is_subjective"  # Column name indicating if the question is subjective
