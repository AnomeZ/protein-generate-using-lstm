from torch import nn

#词汇表
vocab=['-','A','R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V','Z','B']#这里用列表，因为列表是有顺序的结构，索引能和字符对应上
vocab_size = len(vocab)
#字符到索引、索引到字符的映射
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char={idx: char for idx, char in enumerate(vocab)}

#模型架构和参数
embedding_dim = 16
hidden_dim = 256
num_layers = 8

class ProteinGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(ProteinGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out
