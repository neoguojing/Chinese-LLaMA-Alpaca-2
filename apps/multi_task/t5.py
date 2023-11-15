# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


from torch.utils.data import Dataset, DataLoader
import torch

if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')

tokenizer = None
model = None
# 载入预训练的T5模型和tokenizer
def load():
    model_name = 'google/mt5-base'  # 使用T5的基础版本
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir="../../model/mt5",model_max_length=512)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,cache_dir="../../model/mt5")
    model.to(device)

# 定义问题生成任务的数据集类
class QuestionGenerationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        source_text = example['context']
        target_text = example['question']
        source_tokens = tokenizer.encode(source_text, truncation=True, padding='max_length', max_length=512)
        target_tokens = tokenizer.encode(target_text, truncation=True, padding='max_length', max_length=32)
        return {'input_ids': source_tokens, 'labels': target_tokens}

# 准备示例数据
train_data = [
    {'context': '这是一个示例上下文1。', 'question': '请提一个问题。'},
    {'context': '这是一个示例上下文2。', 'question': '请提另一个问题。'},
    # 添加更多的训练数据...
]

# 创建数据集和数据加载器
train_dataset = QuestionGenerationDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 定义训练参数
epochs = 3
learning_rate = 1e-4

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

def do_train():
    # 设置模型为训练模式
    model.train()
    # 微调模型
    for epoch in range(epochs):
        for batch in train_loader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} Loss: {loss.item()}")

def do_eval():
    # 使用微调后的模型生成问题
    context = '''翻译为中文:"You should definitely watch 'One Piece', it is so good, you will love the comic book'''
    print(context)
    input_ids = tokenizer.encode(context, return_tensors='pt',padding=True)
    output = model.generate(input_ids.to(device),max_new_tokens=512)
    generated_question = tokenizer.decode(output[0].cpu(), skip_special_tokens=True)
    print("生成的问题：", generated_question)

# if __name__ == '__main__':
#     do_eval()