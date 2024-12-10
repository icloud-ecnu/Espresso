import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
current_dir = Path(__file__).resolve()
# print("model",sys.path)
parent_dir = current_dir.parent.parent.parent
print(parent_dir)
sys.path.append(str(parent_dir))
from Espresso import GPUAllocator
from Espresso.utils import Model
import json

if __name__ == "__main__":
    with open('./open_llama_3b_v2.json', 'r') as f:
        model = Model(**json.load(f))

    for seqlen in range(1000, 9001, 1000):
        model.seq_len = seqlen
        print(model)  # 确保打印修改后的状态
        min_cost = GPUAllocator.gpu_allocator(model)
        print(f"seqlen = {seqlen} min_cost = {min_cost}")

    with open("./result.json","w") as f:
        f.write(json.dumps(min_cost))
"""

"""