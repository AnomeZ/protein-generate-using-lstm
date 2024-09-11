import matplotlib.pyplot as plt

def visualize(list_to_vis, lens_to_vis):

    assert len(list_to_vis) == lens_to_vis, "列表长度应与lens_to_vis一致"


    # 创建画布和轴
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, lens_to_vis + 1), list_to_vis, label='loss', color='blue')  # 绘制训练准确率

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.title('loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('loss')

    # 设置x轴的刻度为1到epochs
    interval= lens_to_vis // 20

    plt.xticks(range(1, lens_to_vis + 1, interval), rotation=45)
    # 显示网格
    plt.grid(True)

    # 显示图表
    plt.show()
