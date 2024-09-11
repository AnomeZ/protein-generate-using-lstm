# protein-generate-using-lstm
a project using lstm to generate protein sequence

使用了lstm模型去生成蛋白质序列，数据集使用的是从uniprot上下载的fasta文件。
包含以下文件：
data：包含源文件fasta格式，数据处理脚本fasta_to_csv.py，清洗后的data_xx.py，以及padding后的data_padding_xx.py，后缀代表的是蛋白质种类
model_state：用于保存训练后的模型数据
model.py: 模型定义和词表建立，使用的是lstm模型，词表是手动创建的
my_train.py: 模型训练脚本
my_sample.py:模型生成脚本
sequence_match.py: 使用levenshein对生成序列进行简单的相似度分析
Sequence_Alignment.py: 使用Sequence Alignment方法对生成序列进行相似度分析
