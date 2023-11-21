import sys
import os
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将当前package的父目录作为顶层package的路径
top_package_path = os.path.abspath(os.path.join(current_dir, ".."))

# 将顶层package路径添加到sys.path
sys.path.insert(0, top_package_path)

model_root = os.path.join(top_package_path, "model")
print("model root:",model_root)


message = {
    "from":"keyboard",
    "to":None,
    "format":"text",
    "data": ""
}