import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
import vis#自己写的一个可视化函数


from model import char_to_idx, ProteinGenerator, vocab_size, embedding_dim, hidden_dim, num_layers

#打开文件
file_home='data/'
state_home='model_state2/'
data_file_name=('data_padding_CuSOD.csv')

data = pd.read_csv(file_home+data_file_name, header=None)
d = data[0]

# 读取数据并进行预处理
sequences = []
for sequence in d:
    sequences.append(sequence)
print('序列总数：',len(sequences))

# #给出现的氨基酸序列记个数
# Counter=Counter()
# for seq in sequences:
#     for char in seq:
#         Counter[char]+=1
# print(Counter)

# 将序列转换为索引序列
indexed_sequences = [[char_to_idx[char] for char in seq] for seq in sequences]

lengths = [len(seq) for seq in sequences]

# 转换为张量
seq_tensor = torch.tensor(indexed_sequences, dtype=torch.long)

# 定义数据集类
class ProteinDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

# 创建数据加载器
dataset = ProteinDataset(seq_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

#创建模型实例
model = ProteinGenerator(vocab_size, embedding_dim, hidden_dim, num_layers)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()#交叉熵损失函数
#criterion=nn.CosineSimilarity(dim=1, eps=1e-6)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 添加 dropout 层以防止过拟合
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    model.fc
)

# 将模型和数据移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('使用设备:', device)
model.to(device)

# 训练模型
num_epochs = 20
loss_list=[]
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')):
        batch = batch.to(device)

        optimizer.zero_grad()
        outputs = model(batch)

        loss = criterion(outputs.reshape(-1, vocab_size), batch.reshape(-1))
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    loss_list.append(loss.item())

    # 每训练完一轮保存一次模型，嫌占用空间太多可以直接删掉这一行
    torch.save(model.state_dict(), state_home+f'protein_generator_model_epoch_{epoch + 1}.pth')

    #损失小于0.003时可以提前结束训练
    if loss.item()<=0.003:
        break
vis.visualize(loss_list,lens_to_vis=num_epochs)
# 保存最终模型
torch.save(model.state_dict(), state_home+'protein_generator_model_final.pth')