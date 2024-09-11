import numpy as np
import torch
from torch import nn

import model as m
from model import ProteinGenerator, embedding_dim, vocab_size, hidden_dim, num_layers


#参数设置
maxlen = 512 #要生成的序列长度
num_to_generate=10 #要生成的序列数量
temperature = 0.5#引入温度设置

# 读取模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinGenerator(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)

# 加载模型
state_dict = torch.load('model_state/protein_generator_model_epoch_4.pth', map_location=device)

# 手动调整权重
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('fc.1'):
        new_key = key.replace('fc.1', 'fc')
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# 加载调整后的权重
model.load_state_dict(new_state_dict)

model.eval()

print("Model layers dimensions:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")


# 提供一个小部分的蛋白质序列开头
seed_sequence = "MSTAIALNPVTFPLSGTRLIEASAGTGKTYTIAALYVR"
print("Seed sequence:", seed_sequence)

# 将种子序列转换为索引序列
indexed_seed_sequence = [m.char_to_idx[char] for char in seed_sequence]

# 设置最大序列长度

# 填充序列
padded_seed_sequence = indexed_seed_sequence + [0] * (maxlen - len(indexed_seed_sequence))
assert len(padded_seed_sequence) == maxlen, f"Expected length {maxlen}, but got {len(padded_seed_sequence)}"

# 转换为张量
input_tensor = torch.tensor([padded_seed_sequence], dtype=torch.long).to(device)

# input_tensor = np.random.randint(0,vocab_size,(1,maxlen))
# input_tensor = torch.LongTensor(input_tensor).to(device)


assert input_tensor.shape == (1, maxlen), f"Expected shape (1, {maxlen}), but got {input_tensor.shape}"

generated_sequences = []
for i in range(num_to_generate):
    with torch.no_grad():
        generated_sequence = []
        for _ in range(maxlen):  # 生成与 maxlen 长度相同的序列
            output = model(input_tensor)
            probs = torch.softmax(output.reshape(-1)/temperature, dim=0)#经过一层softmax，将其作为权重

            next_char_idx = torch.multinomial(probs, num_samples=1).item()
            #TODO
            generated_sequence.append(m.idx_to_char[next_char_idx])
            input_tensor = torch.cat([input_tensor[:, 1:], torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)

    generated_protein_sequence = ''.join(generated_sequence)
    generated_sequences.append(generated_protein_sequence)




for generated_protein_sequence in generated_sequences:
    print('generated sequence:'+seed_sequence+generated_protein_sequence)

with open('generated_sequences.txt', 'w') as f:
    for generated_sequence in generated_sequences:
        f.write(seed_sequence+generated_sequence+'\n')