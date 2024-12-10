from pathlib import Path
import os
import math

PP_SIZES = [x for x in range(1,24)]
DP_SIZES = [1,2,4,8]
TP_SIZES = [1,2,4,8]
BASE_DIR = Path(__file__).parent
EXPR2_PLAN_DIR = os.path.join(Path(__file__).parent,"experiments","expr2","data_plan")
GPU_CONFIG_FILE = "gpu_config.json"
TOPK_GPU_CONFIG_FILE = "topk_gpu_config.json"
PP_COMM = [0 for _ in range(PP_SIZES[-1] + 1)]

inf = 10000000000
unitResource = 20
unitMem = 8

RESOURCE = 4 * 6 * math.ceil(160 / unitResource)
MEMORY = 4 * 6 * math.ceil(48 / unitMem)