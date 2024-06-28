devices="0,1,2,3"

cleanup(){
    echo "Cleaning up..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait
    echo "All child processes have been terminated."
}

IFS=',' read -r -a deviceArray <<< "$devices"
index=0
trap cleanup SIGINT SIGTERM
pids=()

for element in "${deviceArray[@]}"
do
    export CUDA_VISIBLE_DEVICES=$element
    port_num=$((3660+index))
    python -m vllm.entrypoints.openai.api_server --model MiniCPM-2B-sft-bf16 --dtype auto --api-key s2l --port $port_num --host 127.0.0.1 --trust-remote-code&
    pids+=($!)
    index=$((index+1))
done

wait