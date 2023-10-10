from modelscope import snapshot_download

# model_dir = snapshot_download("qwen/Qwen-7B", revision = 'v1.1')
model_dir = snapshot_download("Qwen/Qwen-7B-Chat", revision = 'v1.1')
print(model_dir)