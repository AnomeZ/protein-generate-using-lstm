
#代码参考：https://zhuanlan.zhihu.com/p/599474543
#这是一种蛋白质序列比对算法Sequence Alignment
import math
import numpy as np
from tqdm import tqdm
# %% 初始化
def str_to_list(a: str, b: str):
    """
    将序列字符串转换为单个字符列表
    a: 序列一
    b: 序列二
    """
    return list(a), list(b)


def ini_matrix(l1: list, l2: list, gap):
    """
    初始化罚分矩阵
    l1: 序列一列表
    l2: 序列二列表
    gap: 空位得分
    """
    # 获取序列长度构建初始矩阵
    n1 = len(l1)
    n2 = len(l2)
    score_matrix = np.zeros((n1 + 1, n2 + 1))
    # 返回矩阵
    return score_matrix


# %% 计分
def score(matrix: np.array, l1: list, l2: list, match, mismatch, gap):
    """
    计算矩阵得分
    matrix: 初始化的矩阵
    l1: 序列一列表
    l2: 序列二列表
    match: 匹配得分
    mismatch: 不匹配时得分
    gap: 空位得分
    """
    # 循环计分
    for i in range(1, len(l1) + 1):
        for j in range(1, len(l2) + 1):
            # 计算三类分值
            from_left = matrix[i][j - 1] + gap  # 从左到右空位
            from_above = matrix[i - 1][j] + gap  # 从上到下空位
            if l1[i - 1] == l2[j - 1]:  # 对角线
                from_diag = matrix[i - 1][j - 1] + match  # 匹配
            else:
                from_diag = matrix[i - 1][j - 1] + mismatch  # 不匹配
            # 比较并赋分
            matrix[i][j] = max(from_left, from_above, from_diag, 0)
    return matrix


# %% 回溯
def trace_back(res: np.array, l1: list, l2: list, match, mismatch, gap):
    """
    回溯矩阵获得匹配结果索引
    res: 结果矩阵
    l1: 序列一列表
    l2: 序列二列表
    match: 匹配得分
    mismatch: 不匹配时得分
    gap: 空位得分
    """
    path = []  # 最终所有路径
    # 找到矩阵中的最大值并入栈
    i = np.where(res == np.max(res))[0][0]
    j = np.where(res == np.max(res))[1][0]
    m_stack = [(i, j)]  # 主栈
    a_stack = []  # 辅助栈
    while m_stack:  # 当主栈非空时
        # 检查是否到终点
        row = m_stack[-1][0]
        col = m_stack[-1][1]
        if res[row - 1][col - 1] == res[row - 1][col] == res[row][col - 1] == 0:
            # 所有邻居都是0，到终点，存储索引路径，依次弹出栈顶
            path.append(m_stack.copy())
            m_stack.pop()
        else:
            if len(m_stack) > len(a_stack):
                # 检查主辅栈长度是否一致，不一致则添加新邻居
                a_stack.append([])
                if l1[row - 1] == l2[col - 1] and res[row][col] == res[row - 1][col - 1] + match:
                    a_stack[-1].append((row - 1, col - 1))
                elif res[row][col] == res[row - 1][col - 1] + mismatch:
                    a_stack[-1].append((row - 1, col - 1))
                if res[row][col] == res[row - 1][col] + gap:
                    a_stack[-1].append((row - 1, col))
                if res[row][col] == res[row][col - 1] + gap:
                    a_stack[-1].append((row, col - 1))
            # 检测辅栈栈顶列表是否为空，不空则可以访问邻居
            elif a_stack[-1] != []:
                m_stack.append(a_stack[-1].pop())
            # 辅助栈为空，则同时退栈一个
            elif a_stack[-1] == []:
                a_stack.pop()
                m_stack.pop()
    return path

# %% 主函数
def sw(seq1: str, seq2: str, match, mismatch, gap):
    """
    主函数
    seq1: 第一条序列
    seq2: 第二条序列
    match: 匹配得分
    mismatch: 不匹配时得分
    gap: 空位得分
    plot: 逻辑值,是否画出得分矩阵,默认不画出
    plot_val: 逻辑值,是否将矩阵数值绘制在图中,默认为False
    """
    # 获取列表
    l1, l2 = str_to_list(seq1, seq2)
    # 初始化打分矩阵
    score_matrix = ini_matrix(l1, l2, gap)
    # 为矩阵赋分
    res_matrix = score(score_matrix, l1, l2, match, mismatch, gap)
    return np.max(res_matrix)

# %% 应用
def compare_sequence(seq1: str, seq2: str):
    start_second = sw(seq1, seq2, match=9, mismatch=-3, gap=-2)
    start = sw(seq1, seq1, match=9, mismatch=-3, gap=-2)
    second = sw(seq2, seq2, match=9, mismatch=-3, gap=-2)

    find_score = start_second / (math.sqrt(start) * math.sqrt(second))
    find_score = round(find_score, 5)
    return find_score


with open('generated_sequences.txt', 'r') as f1:
    generated_sequences = f1.readlines()
with open('data/data_mdh.csv', 'r') as f2:
    src_sequences = f2.readlines()



with open('alignment_result.txt', 'w') as f3:
    for i in range(len(generated_sequences)):
        best_score = 0
        best_sequence = None
        for j in tqdm(range(len(src_sequences))):
            s=compare_sequence(generated_sequences[i], src_sequences[j])
            if s > best_score:
                best_score = s
                best_sequence = src_sequences[j]

        print(generated_sequences[i] + 'best sequence:' + best_sequence + f"score:{best_score:.4f}\n")
        f3.write(generated_sequences[i] + 'best sequence:' + best_sequence + f"score:{best_score:.4f}\n")
