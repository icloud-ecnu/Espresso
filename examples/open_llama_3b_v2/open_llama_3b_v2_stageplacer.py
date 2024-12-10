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
from Espresso import StagePlacer
from Espresso.utils import Model,load_gpus_from_config,test_gpu_detail,profile_latency,Plan
import json

if __name__ == "__main__":

    # for seqlen in range(1000, 9001, 1000):
    #     model.seq_len = seqlen
    #     print(model)  # 确保打印修改后的状态
    #     min_cost = GPUAllocator.gpu_allocator(model)
    #     print(f"seqlen = {seqlen} min_cost = {min_cost}")

    # with open("./result.json","w") as f:
    #     f.write(json.dumps(min_cost))


    gpus = load_gpus_from_config()
    with open('./open_llama_3b_v2.json', 'r') as f:
        model = Model(**json.load(f))
    model.dp_size = 2
    print(f"mainmodel = {model}")
    # 2 3 0 1
    gpu_config = [gpus[0],gpus[2],gpus[3]]
    test_gpu_detail(
        model=model,
        gpus=gpus,
    )
    computeTime = profile_latency(
        model=model,
        gpus=gpus,
    )
    plan = Plan(0, gpu_config, len(gpu_config), model.dp_size, model.tp_size)
    BestConfig = StagePlacer.StageGPUmapper(plan,model)
    print(f"ans = {BestConfig}") # test
"""

"""