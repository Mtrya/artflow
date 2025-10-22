from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
print(config.architectures)