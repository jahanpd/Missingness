export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export XLA_PYTHON_CLIENT_PREALLOCATE=true
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.70
python benchmark_anzscts.py --path=~/CardiacFlask/full_data.csv --imputation None simple iterative miceforest --save
