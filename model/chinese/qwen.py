from modelscope import snapshot_download

# model_dir = snapshot_download("qwen/Qwen-7B", revision = 'v1.1',cache_dir=".")
model_dir = snapshot_download("qwen/Qwen-7B-Chat", revision = 'v1.1',cache_dir=".")
print(model_dir)