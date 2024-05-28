from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/mnt/data/user/tc_agi/zzh6025/Meta-Llama-3-8B-Instruct/", trust_remote_code=True)
outputs = llm.generate(prompts, sampling_params)
print(outputs)