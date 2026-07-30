TOTAL_BATCH_SIZE = 524288  # in tokens, 2 ** 19 (a "nice" number)
B = 2  # micro batch size
T = 1024  # sequence length

MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 715  # gpt3 used 375e6 warmup tokens.  375e6 / TOTAL_BATCH_SIZE = 715
MAX_STEPS = 19073  # TOTAL_TOKENS (10e9) / TOTAL_BATCH_SIZE = 19073

WEIGHT_DECAY = 0.1

EVAL_INTERVAL = 100
