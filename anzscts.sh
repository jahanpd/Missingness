# export CUDA_VISIBLE_DEVICES=1
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform

ANZSCTS=path_to_secure_vpn

python benchmark_anzscts.py --path=$ANZSCTS --imputation None simple iterative miceforest --save
