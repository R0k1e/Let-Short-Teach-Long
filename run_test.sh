pip uninstall vllm
mkdir -p ~/.config/vllm/nccl/cu12/
cp /mnt/data/user/tc_agi/zzh6025/libnccl.so.2.18.1 ~/.config/vllm/nccl/cu12/     #添加依赖
pip install vllm

python run_test.py