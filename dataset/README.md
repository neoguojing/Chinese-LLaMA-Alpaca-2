# format

## for llama origin model 
### https://github.com/facebookresearch/llama-recipes/blob/main/docs/Dataset.md

```
from datasets import load_dataset

dataset = load_dataset("samsum")
```

## pt
- 
## spt
https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/data/alpaca_data_zh_51k.json
```
[
  {
    "instruction": "我们如何在日常生活中减少用水？",
    "input": "",
    "output": "1. 使用节水装置，如节水淋浴喷头和水龙头。 \n2. 使用水箱或水桶收集家庭废水，例如洗碗和洗浴。 \n3. 在社区中提高节水意识。 \n4. 检查水管和灌溉系统的漏水情况，并及时修复它们。 \n5. 洗澡时间缩短，使用低流量淋浴头节约用水。 \n6. 收集雨水，用于园艺或其他非饮用目的。 \n7. 刷牙或擦手时关掉水龙头。 \n8. 减少浇水草坪的时间。 \n9. 尽可能多地重复使用灰水（来自洗衣机、浴室水槽和淋浴的水）。 \n10. 只购买能源效率高的洗碗机和洗衣机。"
  },
  {
    "instruction": "编辑文章，使其更吸引读者。",
    "input": "自主机器人是计算机控制的机器，被编程执行特定任务而不需要任何人类输入。自主机器人在各个行业中被越来越广泛地应用，从制造业到医疗保健再到安全。",
    "output": "自主机器人是计算机控制的机器，被编程执行特定任务而不需要任何人类输入，从而实现了新的效率、精确度和可靠性水平。自主机器人在各个行业中被越来越广泛地应用，从制造业，它们可以使用精度和一致的质量组装复杂的组件，到医疗保健，可以协助进行医疗测试和处理，再到安全，可以监控大面积地区，保障人们和财产的安全。自主机器人还可以减少在危险或有害环境中的错误和增加安全，在工业流程的检查或维修期间等。由于其多样性，自主机器人将彻底改变我们工作方式的方式，使任务变得更加简单、快速，最终更加愉悦。"
  }
]
```
## c-eval
```
from datasets import load_dataset
dataset=load_dataset(r"ceval/ceval-exam",name="computer_network")

print(dataset['val'][0])
```