import os

ERROR_PREFIX = "@|@|@INTERNAL_ERROR@|@|@"  # error prefix for internal errors

DEFAULT_DATASET_NAME = 'penguin_scrolls'
DATASET_NAME = os.environ.get('PENGUIN_SCROLLS', DEFAULT_DATASET_NAME)
DEFAULT_SPLIT = 'test'
PENGUIN_EVAL_MODEL = os.environ.get('PENGUIN_EVAL_MODEL', 'gpt-4o')

INPUT_COL = 'input'
KEY_COL = 'input_md5'
QUESTION_COL = 'prompt'
ANSWER_COL = 'output'
RESPONSE_COL = 'response'
EVAL_RESULT_COL = 'result'
SCORE_COL = 'score'
ERROR_COL = 'error'
TRUNCATED_COL = 'truncated'
