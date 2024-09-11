#测试文件，内容和主程序无关
import numpy as np
import model as m
import torch

from model import idx_to_char

temperature=0.5
maxlen=512
x=torch.randn(1,512,23)
x1=x.reshape(-1)
print(x1)
print(x1.shape)
# sequence=[]
# for i in range(maxlen):
#     a=torch.softmax(x1[i]/temperature,dim=0)
#     idx=torch.multinomial(a,num_samples=1).item()
#     p=a[idx]
#     if idx ==0:
#         idx = torch.multinomial(a, num_samples=1).item()
#     elif p>=0.01:
#         char=idx_to_char[idx]
#     else:
#         break
#     sequence.append(char)
#     print(char)