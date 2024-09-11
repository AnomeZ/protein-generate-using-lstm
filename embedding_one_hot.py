import torch
import numpy as np
from numpy.ma.core import shape

from frist_try.model import vocab_size

# 定义20种标准氨基酸的词汇表
vocab = range(vocab_size)

# 创建一个映射字典，将氨基酸映射到one-hot编码
amino_acid_to_one_hot = {amino_acid: np.eye(len(vocab))[i] for i, amino_acid in enumerate(vocab)}

def tensor_one_hot(str):
    list=[]
    for sequence in str:
        list.append(sequence_to_one_hot(sequence, amino_acid_to_one_hot))
    tensor = torch.tensor(list,dtype=torch.long)

    return tensor


# 定义一个函数，将蛋白质序列转换为one-hot编码
def sequence_to_one_hot(sequence, amino_acid_to_one_hot):
    one_hot_sequence = [amino_acid_to_one_hot[amino_acid] for amino_acid in sequence if
                        amino_acid in amino_acid_to_one_hot]
    return np.array(one_hot_sequence)

#以下为运行尝试
# x=np.random.randint(0,22,(128,809))
# print(x.shape)
# tensor=tensor_one_hot(x)
# print(tensor.shape)
# print('out',tensor)
