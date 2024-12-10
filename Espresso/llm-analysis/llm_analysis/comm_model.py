import copy

from deepspeed.runtime.pipe import schedule
import math

def cal_op(micro_batches, num_stages, stage_id):
    _INSTRUCTION_MAP = {
        schedule.ForwardPass: "F",
        schedule.BackwardPass: "B",
    }
    # print(f"cal_op micro_batches = {micro_batches} num_stages = {num_stages} stage_id = {stage_id}")
    pipe_schedule = schedule.TrainSchedule(micro_batches=micro_batches,
                                        stages=num_stages,
                                        stage_id=stage_id)
    a = []
    for step_cmds in pipe_schedule:
        # For each instruction in the step
        # print(f"step_cmds = {step_cmds}")
        for cmd in step_cmds:
            if type(cmd) not in _INSTRUCTION_MAP:
                continue
            a.append(_INSTRUCTION_MAP[type(cmd)])
    # print(f"a = {a}")
    return a

def get_newop_addC(op):
    new_op = []
    for sublist in op:
        new_sublist = []
        m = len(sublist)
        for i in range(m):
            new_sublist.append(sublist[i])
            if i != m - 1:
                new_sublist.append('C')
        new_op.append(new_sublist)
    return new_op


def cal_pipeline_time(fwd_time, bwd_time, micro_batches,need=False, useNew = False):
    # print(allreduce)
    # print(fwd_time)
    # print(bwd_time)
    num_stages = len(fwd_time)
    op = [cal_op(micro_batches, num_stages, i) for i in range(num_stages)]# op表示每一个stage的Forward和Backward的执行顺序
    cnt = copy.deepcopy(op)
    time = copy.deepcopy(op)
    startTime = copy.deepcopy(op)
    pos = [-1 for _ in range(num_stages)] # pos[i] 表示每一个stage到哪一步了
    for i in range(num_stages):
        for j in range(len(op[i])):
            if op[i][j] == 'F' and i == 0:
                cnt[i][j] = 0  # cnt[i][j] == 0 表示不存在依赖关系
            else:
                cnt[i][j] = 1  # cnt[i][j]表示存在依赖关系
            time[i][j] = 0
            startTime[i][j] = 0
    mx_time = 0
    while sum([len(op[_]) - pos[_] for _ in range(num_stages)]) != num_stages:
        hold = []
        for i in range(len(op)):
            if pos[i] + 1 < len(op[i]) and cnt[i][pos[i] + 1] == 0:
                if pos[i] == -1:
                    hold.append((time[i][pos[i] + 1], i))
                else:
                    # 上一步结束时间和当前这步开始时间
                    hold.append((max(time[i][pos[i]], time[i][pos[i] + 1]), i))
        # 就是有很多个可以处理的，当前选时间最小的
        hold.sort()  # hold 获得下一个应该是哪个stage的哪一步
        id = hold[0][1]  # stage id
        expect_time = hold[0][0]  # 开启的时间，就是我做这个任务他从多久开始
        pos[id] += 1  # 表示这个stage完成了这个task，往后推一个，pos[id]表示当前这个stage完成到哪个地方了，指向的值是已经完成了。
        # print(id,pos[id],expect_time,hold)
        # time[id][pos[id]] 表示id这个stage的pos[id]个task任务的完成时间
        startTime[id][pos[id]] = expect_time
        if op[id][pos[id]] == 'F':
            time[id][pos[id]] = expect_time + fwd_time[id]
        else:
            time[id][pos[id]] = expect_time + bwd_time[id]
        mx_time = max(mx_time, time[id][pos[id]])
        # 我当前在stage为id完成了pos[id]这个任务，那么我应该去更新其他的依赖关系了，如果是fwd，那么下个stage后面可以继续fwd
        if op[id][pos[id]] == 'F' and id < num_stages - 1: # 不是最后一个stage
            for j in range(len(op[id + 1])):
                if cnt[id + 1][j] == 1 and op[id + 1][j] == 'F':
                    cnt[id + 1][j] = 0
                    time[id + 1][j] = time[id][pos[id]]
                    break
        # 我当前在stage为id完成了pos[id]这个任务，那么我应该去更新其他的依赖关系了，如果是fwd最后一个，那么这个stage应该bwd
        elif op[id][pos[id]] == 'F' and id == num_stages - 1: # 是最后一个stage
            for j in range(len(op[id])):
                if cnt[id][j] == 1 and op[id][j] == 'B':
                    cnt[id][j] = 0
                    time[id][j] = time[id][pos[id]]
                    break
        #如果是bwd，就应该继续向前bwd
        if op[id][pos[id]] == 'B' and id > 0: # 不是第一个stage
            for j in range(len(op[id - 1])):
                if cnt[id - 1][j] == 1 and op[id - 1][j] == 'B':
                    cnt[id - 1][j] = 0
                    time[id - 1][j] = time[id][pos[id]]
                    break
    # print(startTime)
    # print(time)
    # print(f"mx_time = {mx_time}")
    # print(op)
    # print(f"time = {time}")
    if need:
        return startTime,time,op
    return mx_time

import random


def initial_config(total_layers, num_stages):
    # 分配每个阶段的层数，确保总数等于total_layers
    config = [total_layers // num_stages for _ in range(num_stages - 1)]
    config.append(total_layers - sum(config))
    random.shuffle(config)
    return config


def neighbor(config):
    idx = random.randint(0, len(config) - 1)
    if idx == 0:
        if config[idx] > 0:
            config[idx] -= 1
            config[idx + 1] += 1
    elif idx == len(config) - 1:
        if config[idx] > 0:
            config[idx] -= 1
            config[idx - 1] += 1
    else:
        if config[idx] > 0:
            if random.randint(0, 1) == 0:
                config[idx] -= 1
                config[idx - 1] += 1
            else:
                config[idx] -= 1
                config[idx + 1] += 1
    return config


def solve(fwd, gradient_accumulation_steps):
    bwd = [x * 2 for x in fwd]
    return cal_pipeline_time(fwd, bwd, gradient_accumulation_steps)


def simulated_annealing(total_layers, num_stages, gradient_accumulation_steps):
    temp = 10.0
    temp_min = 1e-14
    cooling_rate = 0.96
    current_config = initial_config(total_layers, num_stages)
    current_score = solve(current_config, gradient_accumulation_steps)
    best_in_score = 1000000
    while temp > temp_min:
        new_config = neighbor(copy.deepcopy(current_config))
        new_score = solve(new_config,gradient_accumulation_steps)
        if new_score < best_in_score:
            best_in_score = new_score
            best_in_config = new_config
        e = math.exp(min(100, abs(new_score-current_score)/temp))
        if new_score <= current_score or random.random() < 1.0/(1+e):
            current_config = new_config
            current_score = new_score

        temp *= cooling_rate
    return best_in_config, best_in_score



if __name__ == "__main__":
    # fwd = [0.414842,0.413193,0.307763,0.253659,0.286825]
    # bwd = [1.185634,1.185895,0.885842,0.732580,0.798145]

    # fwd = [0.309541,0.361596,0.361144,0.357590,0.285605]
    # bwd = [0.885847,1.033847,1.036607,1.029561,0.797842]

    best_config = None
    best_score = float('inf')
    total = 120
    num_stages = 4
    gradient_accumulation_steps = 16
    for _ in range(30):
        config, score = simulated_annealing(total, num_stages, gradient_accumulation_steps)
        if score < best_score:
            best_score = score
            best_config = copy.deepcopy(config)

    print(f"best_config = {best_config} best_score = {best_score}")
    # std_config = [total // num_stages for x in range(num_stages)]
    # std_score = solve(std_config,gradient_accumulation_steps)
    # print(f"std_config = {std_config} std_score = {std_score}")
    # print(f"{(std_score - best_score) / best_score}")
    config = [30 for x in range(4)] 
    print(solve(config,16))

    # pipe_schedule = schedule.TrainSchedule(micro_batches=4,
    #                                 stages=4,
    #                                 stage_id=1,
    #                                 # fwd_bwd_list = ["F","F","F","B","F","B","B","B"]
    #                                 )
    # for step_cmds in pipe_schedule:
    #     # For each instruction in the step
    #     print(f"step_cmds = {step_cmds}")


    # print(f"ans = {solve(config,4)}")
    # import itertools
    # config = [37,50,30,29]
    # num_points = len(config)
    # all_permutations = list(itertools.permutations(range(num_points)))
    # best = None
    # worst = None
    # score = float('inf')
    # for perm in all_permutations:
    #     fwd = [config[x] for x in perm]
    #     cur = solve(fwd,4)
    #     if cur < score:
    #         score = cur
    #         best = copy.deepcopy(fwd)
    # print(best)