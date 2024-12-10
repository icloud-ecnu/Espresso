"""
test
"""

"""
test open_llama_3b_v2 in 4090-pcie-24gb
python -m analysis train --model_name open_llama_3b_v2 --gpu_name 4090-pcie-24gb --tp_size 1 --pp_size 4 --sp_size 1 --dp_size 1 --gradient_accumulation_steps 4 -b 16 --seq_len 1400  --total_num_tokens 1280 --activation_recomputation 2 --flops_efficiency 0.35 --hbm_memory_efficiency 0.5 --mlp_recompute_gelu True
python3 -m analysis train --model_name open_llama_3b_v2 --gpu_name a100-pcie-40gb --tp_size 1 --pp_size 3 --sp_size 1 --dp_size 1 --gradient_accumulation_steps 4 -b 16 --seq_len 1400  --total_num_tokens 1280 --activation_recomputation 2 --flops_efficiency 0.43 --hbm_memory_efficiency 0.55 --mlp_recompute_gelu True

python -m analysis train --model_name open_llama_3b_v2 --gpu_name a100-pcie-40gb --tp_size 1 --pp_size 3 --dp_size 1 --gradient_accumulation_steps 4 -b 16 --seq_len 1400  --activation_recomputation 2 --flops_efficiency 0.43 --hbm_memory_efficiency 0.55 --mlp_recompute_gelu True --num_layers_per_gpu 13 --first_embedding True --last_embedding True


               task  rank  duration
0      BackwardPass     0  0.002507
1      BackwardPass     1  1.042671
2      BackwardPass     2  0.726257
3      BackwardPass     3  0.933004
4      BackwardPass     4  0.160853
5       ForwardPass     0  0.000535
6       ForwardPass     1  0.376243
7       ForwardPass     2  0.259993
8       ForwardPass     3  0.333798
9       ForwardPass     4  0.077161
"""


cmd = """OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 \
         --nnodes=3 \
         --node_rank=0  \ 
         --master_addr="47.106.80.38" \
         --master_port=12345 \
         train.py \
         --partitions 0,12,16,21,29 \
         --syn \
         --gpuType sequence_fix_mix \
         --syn \
         --seqlen 1400 \
         --baseDir ./experiments/expr2/real_data"""
print(cmd)