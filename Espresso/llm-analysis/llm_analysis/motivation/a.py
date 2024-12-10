distributions = [[12,11,9],[12,9,11],[9,11,12],[9,12,11],[11,12,9],[11,9,12]]


for distribution in distributions:
    distribution[0] += 1
    distribution[2] += 2
    for i in range(1,3):
        distribution[i] += distribution[i-1]
    partition_config ="0," +  ",".join(map(str, distribution))
    
    cmd = f"OMP_NUM_THREADS=64 torchrun --standalone --nproc_per_node=3 train.py --partitions {partition_config} --syn --gpuType motivation3 --syn --seqlen 2000 --baseDir ./experiments/motivation/juchiyunData --modelName llama_1.3b"
    print(cmd)