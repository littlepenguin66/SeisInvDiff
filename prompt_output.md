Project Path: SeisInvDiff

I'd like you to generate a high-quality README file for this project, suitable for hosting on GitHub. Analyze the codebase to understand the purpose, functionality, and structure of the project. 

Source Tree:
```
SeisInvDiff
├── data
│   ├── label
│   ├── seg_eage_salt_data
│   ├── sgy_data
│   │   ├── clean
│   │   └── noise
│   └── feature
├── prompt_output.md
├── pictures
│   ├── pre
│   └── after
├── LICENSE
├── README.md
├── templates
│   ├── reverse-engineering-ctf-solver.hbs
│   ├── binary-exploitation-ctf-solver.hbs
│   ├── clean-up-code.hbs
│   ├── document-the-code.hbs
│   ├── fix-bugs.hbs
│   ├── write-github-pull-request.hbs
│   ├── find-security-vulnerabilities.hbs
│   ├── web-ctf-solver.hbs
│   ├── write-git-commit.hbs
│   ├── cryptography-ctf-solver.hbs
│   ├── claude-xml.hbs
│   ├── refactor.hbs
│   ├── improve-performance.hbs
│   ├── code2prompt
│   └── write-github-readme.hbs
├── model
│   ├── dncnn.py
│   ├── train.py
│   ├── denoise_test.py
│   └── save_dir
│       ├── model_epoch1.pth
│       ├── model_epoch11.pth
│       ├── model_epoch10.pth
│       ├── model_epoch17.pth
│       ├── model_epoch15.pth
│       ├── model_epoch8.pth
│       ├── model_epoch2.pth
│       ├── model_epoch3.pth
│       ├── model_epoch20.pth
│       ├── model_epoch9.pth
│       ├── model_epoch6.pth
│       ├── model_epoch7.pth
│       ├── model_epoch16.pth
│       ├── model_epoch4.pth
│       ├── model_epoch14.pth
│       ├── model_epoch13.pth
│       ├── model_epoch5.pth
│       ├── model_epoch19.pth
│       ├── model_epoch12.pth
│       └── model_epoch18.pth
└── utils
    ├── GetPatches.py
    ├── sgy_to_npy.py
    ├── SignalProcessing.py
    ├── dataset.py
    ├── Cut_combine.py
    └── __init__.py

```

`/root/SeisInvDiff/model/dncnn.py`:

```py

import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return x - out

if __name__ == '__main__':
    net = DnCNN(1, num_of_layers=17)
    print(net)
```

`/root/SeisInvDiff/model/train.py`:

```py

# -*-coding:utf-8-*-
"""
Created on 2022.3.31
programing language:python
@author:夜剑听雨
"""
from dncnn import DnCNN
import sys
sys.path.append("/root/SeisInvDiff")
from utils.dataset import MyDataset
from utils.SignalProcessing import batch_snr
from torch import optim
import torch.nn as nn
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import os

# 选择设备，有cuda用cuda，没有就用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载网络，图片单通道1，分类为1。
my_net = DnCNN(1, num_of_layers=17)
# 将网络拷贝到设备中
my_net.to(device=device)
# 指定特征和标签数据地址，加载数据集
train_path_x = "data/feature/"
train_path_y = "data/label/"
# 划分数据集，训练集：验证集：测试集 = 8:1:1
full_dataset = MyDataset(train_path_x, train_path_y)
valida_size = int(len(full_dataset) * 0.1)
train_size = len(full_dataset) - valida_size * 2
# 指定加载数据的batch_size
batch_size = 32
# 划分数据集
train_dataset, test_dataset, valida_dataset = torch.utils.data.random_split(full_dataset,
                                                                         [train_size, valida_size, valida_size])
# 加载并且乱序训练数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# 加载并且乱序验证数据集
valida_loader = torch.utils.data.DataLoader(dataset=valida_dataset, batch_size=batch_size, shuffle=True)
# 加载测试数据集,测试数据不需要乱序
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义优化方法
epochs = 20  # 设置训练次数
LR = 0.001   # 设置学习率
optimizer = optim.Adam(my_net.parameters(), lr=LR)
# 定义损失函数
criterion = nn.MSELoss(reduction='sum')  # reduction='sum'表示不除以batch_size

temp_sets1 = []  # 用于记录训练，验证集的loss,每一个epoch都做一次训练，验证
temp_sets2 = []   # # 用于记录测试集的SNR,去噪前和去噪后都要记录


start_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # 开始时间

# 每一个epoch都做一次训练，验证，测试
for epoch in range(epochs):
    # 训练集训练网络
    train_loss = 0.0
    my_net.train()  # 开启训练模式
    for batch_idx1, (batch_x, batch_y) in enumerate(train_loader, 0):  # 0开始计数
        # 加载数据至GPU
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.float32)
        err_out1 = my_net(batch_x)  # 使用网络参数，输出预测结果
        # 计算loss
        loss1 = criterion(err_out1, (batch_x-batch_y))
        train_loss += loss1.item()  # 累加计算本次epoch的loss，最后还需要除以每个epoch可以抽取多少个batch数，即最后的n_count值
        optimizer.zero_grad()  # 先将梯度归零,等价于net.zero_grad(0
        loss1.backward()  # 反向传播计算得到每个参数的梯度值
        optimizer.step()  # 通过梯度下降执行一步参数更新
    train_loss = train_loss / (batch_idx1+1)  # 本次epoch的平均loss

    # 验证集验证网络
    my_net.eval()  # 开启评估/测试模式
    val_loss = 0.0
    for batch_idx2, (val_x, val_y) in enumerate(valida_loader, 0):
        # 加载数据至GPU
        val_x = val_x.to(device=device, dtype=torch.float32)
        val_y = val_y.to(device=device, dtype=torch.float32)
        with torch.no_grad():  # 不需要做梯度更新，所以要关闭求梯度
            err_out2 = my_net(val_x)  # 使用网络参数，输出预测结果
            # 计算loss
            loss2 = criterion(err_out2, (val_x-val_y))
            val_loss += loss2.item()  # 累加计算本次epoch的loss，最后还需要除以每个epoch可以抽取多少个batch数，即最后的count值
    val_loss = val_loss / (batch_idx2+1)
    # 训练，验证，测试的loss保存至loss_sets中
    loss_set = [train_loss, val_loss]
    temp_sets1.append(loss_set)
    # {:.4f}值用format格式化输出，保留小数点后四位
    print("epoch={}，训练集loss：{:.4f}，验证集loss：{:.4f}".format(epoch+1, train_loss, val_loss))

    # 测试集测试网络，采用计算一个batch数据的信噪比(snr)作为评估指标
    snr_set1 = 0.0
    snr_set2 = 0.0
    for batch_idx3, (test_x, test_y) in enumerate(test_loader, 0):
        # 加载数据至GPU
        test_x = test_x.to(device=device, dtype=torch.float32)
        test_y = test_y.to(device=device, dtype=torch.float32)
        with torch.no_grad():  # 不需要做梯度更新，所以要关闭求梯度
            err_out3 = my_net(test_x)  # 使用网络参数，输出预测结果(训练的是噪声)
            # 含噪数据减去噪声得到的才是去噪后的数据
            clean_out = test_x - err_out3
            # 计算网络去噪后的数据和干净数据的信噪比(此处是计算了所有的数据，除以了batch_size求均值)
            SNR1 = batch_snr(test_x, test_y)  # 去噪前的信噪比
            SNR2 = batch_snr(clean_out, test_y)  # 去噪后的信噪比
        snr_set1 += SNR1
        snr_set2 += SNR2
        # 累加计算本次epoch的loss，最后还需要除以每个epoch可以抽取多少个batch数，即最后的count值
    snr_set1 = snr_set1 / (batch_idx3 + 1)
    snr_set2 = snr_set2 / (batch_idx3 + 1)

    # 训练，验证，测试的loss保存至loss_sets中
    snr_set = [snr_set1, snr_set2]
    temp_sets2.append(snr_set)

    # {:.4f}值用format格式化输出，保留小数点后四位
    print("epoch={}，去噪前的平均信噪比(SNR)：{:.4f} dB，去噪后的平均信噪比(SNR)：{:.4f} dB".format(epoch+1, snr_set1, snr_set2))

    # 保存网络模型
    model_name = f'model_epoch{epoch+1}'  # 模型命名
    torch.save(my_net, os.path.join('model/save_dir', model_name+'.pth'))  # 保存整个神经网络的模型结构以及参数

end_time = time.strftime("1. %Y-%m-%d %H:%M:%S", time.localtime())  # 结束时间
# 将训练花费的时间写成一个txt文档，保存到当前文件夹下
with open('训练时间.txt', 'w', encoding='utf-8') as f:
    f.write(start_time)
    f.write(end_time)
    f.close()
print("训练开始时间{}>>>>>>>>>>>>>>>>训练结束时间{}".format(start_time, end_time))  # 打印所用时间

# temp_sets1是三维张量无法保存，需要变成2维数组才能存为txt文件
loss_sets = []
for sets in temp_sets1:
    for i in range(2):
        loss_sets.append(sets[i])
loss_sets = np.array(loss_sets).reshape(-1, 2)  # 重塑形状10*2，-1表示自动推导
# fmt参数，指定保存的文件格式。将loss_sets存为txt文件
np.savetxt('loss_sets.txt', loss_sets, fmt='%.4f')

# temp_sets2是三维张量无法保存，需要变成2维数组才能存为txt文件
snr_sets = []
for sets in temp_sets2:
    for i in range(2):
        snr_sets.append(sets[i])
snr_sets = np.array(snr_sets).reshape(-1, 2)  # 重塑形状10*2，-1表示自动推导
# fmt参数，指定保存的文件格式。将loss_sets存为txt文件
np.savetxt('snr_sets.txt', snr_sets, fmt='%.4f')

# 显示loss曲线
loss_lines = np.loadtxt('./loss_sets.txt')
# 前面除以batch_size会导致数值太小了不易观察
train_line = loss_lines[:, 0] / batch_size
valida_line = loss_lines[:, 1] / batch_size
x1 = range(len(train_line))
fig1 = plt.figure()
plt.plot(x1, train_line, x1, valida_line)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'valida'])
plt.savefig('pictures/after/loss_plot.png', bbox_inches='tight')
plt.tight_layout()

# 显示snr曲线
snr_lines = np.loadtxt('./snr_sets.txt')
De_before = snr_lines[:, 0]
De_after = snr_lines[:, 1]
x2 = range(len(De_before))
fig2 = plt.figure()
plt.plot(x2, De_before, x2, De_after)
plt.xlabel('epoch')
plt.ylabel('SNR')
plt.legend(['noise', 'denoise'])
plt.savefig('pictures/after/snr_plot.png', bbox_inches='tight')
plt.tight_layout()

plt.show()
```

`/root/SeisInvDiff/model/denoise_test.py`:

```py
# -*-coding:utf-8-*-
"""
Created on 2022.5.1
programing language:python
@author:夜剑听雨
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.GetPatches import read_segy_data
from utils.Cut_combine import cut, combine
import torch

# 加载数据
seismic_noise = read_segy_data('../data/sgy_data/fileld.segy')  # 野外地震数据
seismic_block_h, seismic_block_w = seismic_noise.shape
# 数据归一化处理
seismic_noise_max = abs(seismic_noise).max()  # 获取数据最大幅值
seismic_noise = seismic_noise / seismic_noise_max  # 将数据归一化到(-1,1)
# 对缺失的炮集数据进行膨胀填充，并且切分
patch_size = 64
patches, strides_x, strides_y, fill_arr_h, fill_arr_w = cut(seismic_noise, patch_size, patch_size, patch_size)

# 检测是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载模型
model = torch.load('./save_dir/model_epoch20.pth')
model.to(device=device)  # 模型拷贝至GPU
model.eval()  # 开启评估模式
predict_datas = []  # 空列表，用于存放网络预测的切片数据
# 对切片数据进行网络预测
for patch in patches:
    patch = np.array(patch)  # 转换为numpy数据
    patch = patch.reshape(1, 1, patch.shape[0], patch.shape[1])  # 对数据维度进行扩充(批量，通道，高，宽)
    patch = torch.from_numpy(patch)  # python转换为tensor
    patch = patch.to(device=device, dtype=torch.float32)  # 数据拷贝至GPU
    predict_data = model(patch)  # 预测结果
    predict_data = predict_data.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    print(predict_data.shape)
    predict_data = predict_data.squeeze()  # 默认压缩所有为1的维度
    print(predict_data.shape)
    predict_datas.append(predict_data)  # 添加至列表中

# 对预测后的数据进行还原，裁剪
seismic_predict = combine(predict_datas, patch_size, strides_x, strides_y, seismic_block_h, seismic_block_w)
# 数据逆归一化处理
seismic_predict = seismic_predict*seismic_noise_max  # 将数据归一化到(-1,1)
#  显示处理效果
fig1 = plt.figure()
# 三个参数分别为：行数，列数，
ax1 = fig1.add_subplot(1, 3, 1)
ax2 = fig1.add_subplot(1, 3, 2)
ax3 = fig1.add_subplot(1, 3, 3)
# 绘制曲线
ax1.imshow(seismic_noise, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
ax2.imshow(seismic_predict, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
ax3.imshow(seismic_noise-seismic_predict, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
plt.tight_layout()  # 自动调整子图位置
plt.show()

```

`/root/SeisInvDiff/utils/GetPatches.py`:

```py
# -*-coding:utf-8-*-
"""
Created on 2022.3.1
programing language:python
@author:夜剑听雨
"""
import glob
import cv2
import numpy as np
import segyio
import matplotlib.pyplot as plt
import random
import os

def read_segy_data(filename):
    """
    读取segy或者sgy数据，剥离道头信息
    :param filename: segy或者sgy文件的路径
    :return: 不含道头信息的地震道数据
    """
    print("### Reading SEGY-formatted Seismic Data:")
    print("Data file-->[%s]" %(filename))
    with segyio.open(filename, "r", ignore_geometry=True)as f:
        f.mmap()
        data = np.asarray([np.copy(x) for x in f.trace[:]]).T
    f.close()
    return data

def data_augmentation(img, mode=None):
    """
    data augmentation 数据扩充
    :param img: 二维矩阵
    :param mode: 对矩阵的翻转方式
    :return: 翻转后的矩阵
    """
    if mode == 0:
        # original 原始的
        return img
    elif mode == 1:
        # flip up and down 上下翻动
        return np.flipud(img)
    elif mode == 2:
        # 逆时针旋转90度
        return np.rot90(img)
    elif mode == 3:
        #  先旋转90度，在上下翻转
        return np.flipud(np.rot90(img))
    elif mode == 4:
        #  旋转180度
        return np.rot90(img, k=2)
    elif mode == 5:
        # 旋转180度并翻转
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        # 旋转270度
        return np.rot90(img, k=3)
    elif mode == 7:
        # 旋转270度并翻转
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_path, patch_size, stride_x, stride_y, scales):
    """
    对单炮数据进行数据切片，需要先将单炮数据的数据和道头剥离。
    Args:
        file_path:地震道数据的文件路径。
        patch_size:切片数据的大小，都是方形所以高宽一致。
        stride_x:在地震道数据x方向的滑动步长。
        stride_y:在地震道数据y方向的滑动步长。
        scales:输入为列表，对数据进行放缩。
    Returns:返回一系列的小数据块
    """
    shot_data = np.load(file_path)  # 加载npy数据
    time_sample, trace_number = shot_data.shape  # 获取数据大小
    patches = []   # 生成空列表用于添加小数据块
    for scale in scales:  # 遍历数据的缩放方式
        time_scaled, trace_scaled = int(time_sample * scale), int(trace_number * scale)  # 缩放后取整
        shot_scaled = cv2.resize(shot_data, (trace_scaled, time_scaled), interpolation=cv2.INTER_LINEAR) # 获得缩放后的数据，采用双线性插值
        # 数据归一化处理
        shot_scaled = shot_scaled / abs(shot_scaled).max()  # 将数据归一化到(-1,1)
        # 从放缩之后的shot_scaled中提取多个patch
        # 计算x方向滑动步长位置
        s1 = 1
        while (patch_size + (s1-1)*stride_x) <= trace_scaled:
            s1 = s1 + 1
        # python中索引默认0开始，而且左闭右开。patch_size + (n-1)*stride_x就是切片滑动时候的实际位置加1
        # 这里的n算出来大了1
        strides_x = []  # 用于存储x方向滑动步长位置
        x = np.arange(s1-1)  # 生成0~s1-2的序列数字
        x = x + 1  # 将序列变成1~s1-1
        for i in x:
            s_x = patch_size + (i-1)*stride_x  # 计算每一次的步长位置(实际位置加1)
            strides_x.append(s_x)  # 添加到列表
        # 计算y方向滑动步长位置
        s2 = 1
        while (patch_size + (s2-1)*stride_y) <= time_scaled:
            s2 = s2 + 1
        strides_y = []
        y = np.arange(s2-1)
        y = y + 1
        for i1 in y:
            s_y = patch_size + (i1-1)*stride_y
            strides_y.append(s_y)
        #  通过切片的索引位置在数据中提取小patch
        for index_x in strides_y:  # x方向索引是patch的列
            for index_y in strides_x:  # y方向索引是patch的行
                patch = shot_scaled[index_x-patch_size: index_x, index_y-patch_size: index_y]
                patches.append(patch)
    return patches

def data_generator(data_dir, patch_size, stride_x, stride_y, scales):
    """
    对整个目录下的npy文件进行，数据的切片。
    Args:
        data_dir: 文件夹路径
        patch_size:切片数据的大小，都是方形所以高宽一致。
        stride_x:在地震道数据x方向的滑动步长。
        stride_y:在地震道数据y方向的滑动步长。
        scales:输入为列表，对数据进行放缩。
    Returns:总的切片数据
    """
    file_list = glob.glob(os.path.join(data_dir, '*npy'))
    data = []
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i], patch_size, stride_x, stride_y, scales)
        for patch in patches:
            data.append(patch)
    print("获得切片数量：{}".format(len(data)))
    return data
def calculate_patches(time_number, trace_number, blocks, patch_size, stride_x, stride_y, scales):
    """
    计算数据的切片数量
    Args:
        time_number: 单个数据块的时间采样点数
        trace_number: 单个数据块的地震道数
        blocks: 需要切片的数据块的数量
        patch_size: 切片patch的大小
        stride_x:在地震道数据x方向的滑动步长
        stride_y:在地震道数据y方向的滑动步长
        scales:输入为列表，对数据进行放缩
    Returns: 总的切片数据的数量
    """
    sum_patches = 0
    for scale in scales:  # 便利数据的缩放方式
        time_scaled, trace_scaled = int(time_number * scale), int(trace_number * scale)  # 缩放后取整
        # 从放缩之后的shot_scaled中提取多个patch
        # 数据切分收到滑动步长和数据块尺寸的共同影响，先确定数据块滑动步长位置
        n = 1
        while (patch_size + (n - 1) * stride_x) <= trace_scaled:
            n = n + 1
        # python中索引默认0开始，而且左闭右开。patch_size + (n-1)*stride_x就是切片滑动时候的实际位置加1
        # 这里的n算出来大了1
        strides_x = []  # 用于存储x方向滑动步长位置
        x = np.arange(n - 1)  # 生成0~n-2的序列数字
        x = x + 1  # 将序列变成1~n-1
        for i in x:
            s_x = patch_size + (i - 1) * stride_x  # 计算每一次的步长位置(实际位置加1)
            strides_x.append(s_x)  # 添加到列表
        # 计算y方向滑动步长位置
        n = 1
        while (patch_size + (n - 1) * stride_y) <= time_scaled:
            n = n + 1
        strides_y = []
        y = np.arange(n - 1)
        y = y + 1
        for i in y:
            s_y = patch_size + (i - 1) * stride_y
            strides_y.append(s_y)
        numbers = len(strides_y) * len(strides_x) * blocks
        sum_patches += numbers
    return sum_patches
if __name__ == "__main__":

    # 可用calculate_patches函数提前估算切片数量
    # patch_num = calculate_patches(1501, 301, 30, 64, 32, 64, [1])
    # print(patch_num)

    # 对剥离后的数据进行切分
    data_dir1 = "data/sgy_data/noise/"  # 含噪数据文件路径
    data_dir2 = "data/sgy_data/clean/"  # 干净数据文件路径
    patch_size = 64  # 数据块patch的大小
    scales = [1]     # 数据块拉伸的方式
    xs = data_generator(data_dir1, patch_size, 32, 64, scales)  # 含噪和抽稀数据patches，即特征。
    ys = data_generator(data_dir2, patch_size, 32, 64, scales)  # 干净数据patches，即标签。

    # 对标签和样本数据进行随机翻转, len(xs)=len(ys)
    patches_index = range(len(xs))  # 获取标签数据或者样本数据的长度，变成索引
    enhance_number = int(len(xs) * 0.2)    # 数据增强的数量,20%的比例
    enhance_numbers = random.sample(patches_index, enhance_number)  # 从patches_index随机抽取20%个元素
    for k in enhance_numbers:  # 遍历随机抽取出来的索引
        random_number = random.randint(0, 7)  # 产生一个随机数mode: a <= mode <= b
        # 对标签和样本同步随机翻转
        data_augmentation(xs[k], mode=random_number)
        data_augmentation(ys[k], mode=random_number)

    # 保存数据集
    for j in range(len(xs)):
        feature = xs[j]
        label = ys[j]
        noise_name = f'feature{j+1}'
        label_name = f'label{j+1}'
        np.save("data/feature/" + noise_name, feature)
        np.save("data/label/" + label_name, label)
    print(f'一共保存{len(xs)}个训练集地震数据切片！')

    # 查看若干个切片
    c1 = np.load('data/feature/feature81.npy')
    n1 = np.load('data/label/label81.npy')
    c2 = np.load('data/feature/feature1.npy')
    n2 = np.load('data/label/label1.npy')
    fig1 = plt.figure()
    # 三个参数分别为：行数，列数，
    ax1 = fig1.add_subplot(2, 2, 1)
    ax2 = fig1.add_subplot(2, 2, 2)
    ax3 = fig1.add_subplot(2, 2, 3)
    ax4 = fig1.add_subplot(2, 2, 4)
    # 绘制曲线
    ax1.imshow(c1, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
    ax2.imshow(n1, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
    ax3.imshow(c2, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
    ax4.imshow(n2, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
    plt.tight_layout()  # 自动调整子图位置
    plt.show()

```

`/root/SeisInvDiff/utils/sgy_to_npy.py`:

```py
# -*-coding:utf-8-*-
"""
Created on 2022.4.19
programing language:python
@author:夜剑听雨
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from GetPatches import read_segy_data
import os

path = os.path.join('data/sgy_data', 'synthetic.sgy')  # 获取sgy数据路径
# path = 'data/sgy_data/synthetic.sgy'
sgy_data = read_segy_data(path)  # 读取sgy地震数据
print(sgy_data.shape)   # 查看数据尺寸

# 数据共30炮，每一炮都是301道地震记录，数据不含噪声，将数据提取为单炮第地震记录
for i in range(30):
    # 读取干净的炮集
    clean_shot = sgy_data[:, i*301:(i+1)*301]
    # 保存干净的炮集记录
    clean_name = f'clean{i + 1}'  # 给每一个炮集命名，采用format方法
    np.save('data/sgy_data/clean/' + clean_name, clean_shot)  # 设置保存路径

    # 对数据加噪并且保存
    clean_shot_max = abs(clean_shot).max()    # 获取数据最大幅值
    clean_shot = clean_shot / clean_shot_max  # 将数据归一化到(-1,1)
    noise = np.random.random([clean_shot.shape[0], clean_shot.shape[1]])  # 生成幅值为0~1的随机噪声
    rates = [0.05, 0.1, 0.15]  # 设置随机噪声的幅值
    rate = random.sample(rates, 1)  # 产生一个随机数mode: a <= mode <= b
    noise_shot = clean_shot + rate[0] * noise  # 加入数据幅值rate[0]随机噪声
    noise_shot = noise_shot * clean_shot_max  # 逆归一化
    # 保存含噪声的炮集记录
    noise_name = f'noise{i + 1}'
    np.save('data/sgy_data/noise/' + noise_name, noise_shot)

print(f'第{i + 1}个地震数据已经抽稀保存完毕')

# 查看数据
x1 = np.load('data/sgy_data/clean/clean1.npy')
x2 = np.load('data/sgy_data/clean/clean11.npy')
x3 = np.load('data/sgy_data/clean/clean30.npy')

y1 = np.load('data/sgy_data/noise/noise1.npy')
y2 = np.load('data/sgy_data/noise/noise11.npy')
y3 = np.load('data/sgy_data/noise/noise30.npy')

fig1 = plt.figure()
# 三个参数分别为：行数，列数，
ax1 = fig1.add_subplot(2, 3, 1)
ax2 = fig1.add_subplot(2, 3, 2)
ax3 = fig1.add_subplot(2, 3, 3)
ax4 = fig1.add_subplot(2, 3, 4)
ax5 = fig1.add_subplot(2, 3, 5)
ax6 = fig1.add_subplot(2, 3, 6)
# 绘制曲线gray
ax1.imshow(x1, cmap=plt.cm.seismic, interpolation='nearest', aspect=0.25, vmin=-0.5, vmax=0.5)
ax2.imshow(x2, cmap=plt.cm.seismic, interpolation='nearest', aspect=0.25, vmin=-0.5, vmax=0.5)
ax3.imshow(x3, cmap=plt.cm.seismic, interpolation='nearest', aspect=0.25, vmin=-0.5, vmax=0.5)
ax4.imshow(y1, cmap=plt.cm.seismic, interpolation='nearest', aspect=0.25, vmin=-0.5, vmax=0.5)
ax5.imshow(y2, cmap=plt.cm.seismic, interpolation='nearest', aspect=0.25, vmin=-0.5, vmax=0.5)
ax6.imshow(y3, cmap=plt.cm.seismic, interpolation='nearest', aspect=0.25, vmin=-0.5, vmax=0.5)
plt.tight_layout()  # 自动调整子图位置
plt.show()
plt.savefig('pictures/after/clean_noise.png')
```

`/root/SeisInvDiff/utils/SignalProcessing.py`:

```py
# -*-coding:utf-8-*-
"""
Created on 2022.3.5
programing language:python
@author:夜剑听雨
"""
import numpy as np
import math
from scipy import signal

def compare_SNR(recov_img, real_img):
    """
    计算信噪比
    :param recov_img:重建后或者含有噪声的数据
    :param real_img: 干净的数据
    :return: 信噪比
    """
    real_mean = np.mean(real_img)
    tmp1 = real_img - real_mean
    real_var = sum(sum(tmp1*tmp1))

    noise = real_img - recov_img
    noise_mean = np.mean(noise)
    tmp2 = noise - noise_mean
    noise_var = sum(sum(tmp2*tmp2))

    if noise_var == 0 or real_var==0:
      s = 999.99
    else:
      s = 10*math.log(real_var/noise_var, 10)
    return s
def batch_snr(de_data, clean_data):
    """
    计算一个batch的平均信噪比
    :param de_data: 去噪后的数据
    :param clean_data: 干净的数据
    :return: 一个batch的平均信噪比
    """
    De_data = de_data.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    Clean_data = clean_data.data.cpu().numpy()
    SNR = 0
    for i in range(De_data.shape[0]):
        De = De_data[i, :, :, :].squeeze()  # 默认压缩所有为1的维度
        Clean = Clean_data[i, :, :, :].squeeze()
        SNR += compare_SNR(De, Clean)
    return SNR / De_data.shape[0]

def mse(signal, noise_data):
    """
    计算均方误差
    Args:
        signal: 信号
        noise_data: 含噪声数据
    Returns:均方误差
    """
    signal = np.array(signal)
    noise_data = np.array(noise_data)
    m = np.sum((signal - noise_data) ** 2)  # numpy可以并行运算
    m = m / m.size  # mse.size输出矩阵的元素个数
    return m

def psnr(signal, noise_data):
    """
    计算峰值信噪比
    Args:
        signal: 信号
        noise_data: 含噪声数据
    Returns:峰值信噪比
    """
    signal = np.array(signal)
    noise_data = np.array(noise_data)
    psnr = 2 * 10 * math.log10(abs(signal.max()) / np.sqrt(np.sum((signal - noise_data) ** 2) / noise_data.size))
    return psnr

def fft_spectrum(Signal, SampleRate):
    """
    计算一维信号的傅里叶谱
    :param Signal: 一维信号
    :param SampleRate: 采样率，一秒内的采样点数
    :return: 傅里叶变换结果
    """
    fft_len = Signal.size  # 傅里叶变换长度
    # 原函数值的序列经过快速傅里叶变换得到一个复数数组，复数的模代表的是振幅，复数的辐角代表初相位
    SignalFFT = np.fft.rfft(Signal) / fft_len  # 变换后归一化处理
    SignalFreqs = np.linspace(0, SampleRate/2, int(fft_len/2)+1)  # 生成频率区间
    SignalAmplitude = np.abs(SignalFFT) * 2   # 复数的模代表的是振幅
    return SignalFreqs, SignalAmplitude

# 巴沃斯低通滤波器
def butter_lowpass(cutoff, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    normal_cutoff = cutoff / rate
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(noise_data, cutoff, sample_rate, order=4):
    """
    低通滤波器
    :param noise_data: 含噪声数据
    :param cutoff: 低通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_lowpass(cutoff, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data

# 巴沃斯带通滤波器
def butter_bandpass(lowcut, highcut, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    low = lowcut / rate
    high = highcut / rate
    b, a = signal.butter(order, [low, high], btype='bandpass', analog=False)
    return b, a

def bandpass_filter(noise_data, lowcut, highcut, sample_rate, order=4):
    """
    带通滤波器
    :param noise_data: 含噪声数据
    :param lowcut: 带通滤波的最小值
    :param higtcut: 带通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_bandpass(lowcut, highcut, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data
# 巴沃斯高通滤波器
def butter_highpass(cutup, sample_rate, order=4):
    # 设置滤波器参数
    rate = sample_rate * 0.5
    normal_cutup = cutup / rate
    b, a = signal.butter(order, normal_cutup, btype='high', analog=False)
    return b, a

def highpass_filter(noise_data, cutup, sample_rate, order=4):
    """
    低通滤波器
    :param noise_data: 含噪声数据
    :param cutoff: 低通滤波的最大值
    :param sample_rate: 数据采样率
    :param order: 滤波器阶数，默认为4
    :return: 滤波后的数据
    """
    b, a = butter_highpass(cutup, sample_rate, order=order)
    clear_data = signal.filtfilt(b, a, noise_data)
    return clear_data

# 一维信号的中值滤波器
# python的中值滤波函数对数组的维数要求严格，打个比方你用维数为（200，1）的数组当输入，不行！
# 必须改成（200，才会给你滤波。
def mide_filter(x,kernel_size=5):
    """
    中值滤波器
    :param x: 一维信号
    :param kernel_size: 滤波器窗口，默认为5
    :return: 中值滤波后的数据
    """
    x1 = x.reshape(x.size)
    y = signal.medfilt(x1, kernel_size=kernel_size)
    return y

def fk_spectra(data, dt, dx, L=6):
    """
    f-k(频率-波数)频谱分析
    :param data: 二维的地震数据
    :param dt: 时间采样间隔
    :param dx: 道间距
    :param L: 平滑窗口
    :return: S(频谱结果), f(频率范围), k(波数范围)
    """
    data = np.array(data)
    [nt, nx] = data.shape  # 获取数据维度
    # 计算nk和nf是为了加快傅里叶变换速度,等同于nextpow2
    i = 0
    while (2 ** i) <= nx:
        i = i + 1
    nk = 4 * 2 ** i
    j = 0
    while (2 ** j) <= nt:
        j = j + 1
    nf = 4 * 2 ** j
    S = np.fft.fftshift(abs(np.fft.fft2(data, (nf, nk))))  # 二维傅里叶变换
    H1 = np.hamming(L)
    # 设置汉明窗口大小，汉明窗的时域波形两端不能到零，而海宁窗时域信号两端是零。从频域响应来看，汉明窗能够减少很近的旁瓣泄露
    H = (H1.reshape(L, -1)) * (H1.reshape(1, L))
    S = signal.convolve2d(S, H, boundary='symm', mode='same')  # 汉明平滑
    S = S[nf // 2:nf, :]
    f = np.arange(0, nf / 2, 1)
    f = f / nf / dt
    k = np.arange(-nk / 2, nk / 2, 1)
    k = k / nk / dx
    return S, k, f



```

`/root/SeisInvDiff/utils/dataset.py`:

```py
# -*-coding:utf-8-*-
"""
Created on 2022.3.14
programing language:python
@author:夜剑听雨
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    # 构造函数
    def __init__(self, feature_path, label_path):
        super(MyDataset, self).__init__()
        self.feature_paths = glob.glob(os.path.join(feature_path, '*.npy'))
        self.label_paths = glob.glob(os.path.join(label_path, '*.npy'))

    # 返回数据集大小
    def __len__(self):
        return len(self.feature_paths)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        feature_data = np.load(self.feature_paths[index])
        label_data = np.load(self.label_paths[index])
        feature_data = torch.from_numpy(feature_data)  # numpy转成张量
        label_data = torch.from_numpy(label_data)
        feature_data.unsqueeze_(0)  # 增加一个维度128*128 =>1*128*128
        label_data.unsqueeze_(0)
        return feature_data, label_data

if __name__ == "__main__":

    feature_path = "..\\data\\feature\\"
    label_path = "..\\data\\label\\"
    seismic_dataset = MyDataset(feature_path, label_path)
    train_loader = torch.utils.data.DataLoader(dataset=seismic_dataset,
                                               batch_size=32,
                                               shuffle=True)
    # Img = train_loader.numpy().astype(np.float32)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print('Dataset size:', len(seismic_dataset))
    print('train_loader:', len(train_loader))

```

`/root/SeisInvDiff/utils/Cut_combine.py`:

```py
# -*-coding:utf-8-*-
"""
Created on 2022.5.1
programing language:python
@author:夜剑听雨
"""
# '''
#     1.对原始数据块(arr1)的右方和下方进行填充，使其横向和竖向都可以整除patch(L*L)。
#     2.将切好的patch喂入网络训练后，只取数据的中心部分(L * L),按照顺序拼起来既可以和arr1一样大的数据。
# '''
import numpy as np

def cut(seismic_block, patch_size, stride_x, stride_y):
    """
    :param seismic_block: 地震数据
    :param patch_size: 切片大小
    :param stride_x: 横向切片步长，大小等于patch_size
    :param stride_y: 竖向切片步长，大小等于patch_size
    :return: 按照规则填充后，获得的切片数据(以列表形式存储)，高方向切片数量，宽方向切片数量
    """
    [seismic_h, seismic_w] = seismic_block.shape  # 获取地震数据块的高(seismic_block_h)和宽(seismic_block_w)
    # 对数据进行填充，确保可以完整切片
    # 确定宽方向填充后大小
    n1 = 1
    while (patch_size + (n1 - 1) * stride_x) <= seismic_w:
        # 判断长为patch_size,步长为stride_x在长为seismic_w的时候能滑动多少步
        n1 = n1 + 1
    # 循环结束后计算的patch_size + (n1-1)*stride_x) > seismic_w，在滑动整数步长的时候可以完全覆盖数据
    arr_w = patch_size + (n1 - 1) * stride_x
    # 确定高方向填充后大小
    n2 = 1
    while (patch_size + (n2 - 1) * stride_y) <= seismic_h:
        n2 = n2 + 1
    arr_h = patch_size + (n2 - 1) * stride_y
    # # 对seismic_block数据块的右方和下方进行填充，填充内容为0
    fill_arr = np.zeros((arr_h, arr_w), dtype=np.float32)
    fill_arr[0:seismic_h, 0:seismic_w] = seismic_block
    # 对数据填充后，我们切分的数据是填充后的数据
    # 计算arr_w方向滑动步长位置
    # python中索引默认0开始，而且左闭右开。patch_size + (n-1)*stride_x就是切片滑动时候的实际位置加1
    # 这里的n算出来大了1
    path_w = []  # 用于存储x方向滑动步长位置
    x = np.arange(n1)  # 生成[0~n1-1]的序列数字
    x = x + 1  # 将序列变成[1~n1]
    for i in x:
        s_x = patch_size + (i - 1) * stride_x  # 计算每一次的步长位置(实际位置加1)
        path_w.append(s_x)  # 添加到列表
    number_w = len(path_w)
    path_h = []
    y = np.arange(n2)
    y = y + 1
    for k in y:
        s_y = patch_size + (k - 1) * stride_y
        path_h.append(s_y)
    number_h = len(path_h)
    #  通过切片的索引位置在数据中提取小patch
    cut_patches = []
    for index_x in path_h:  # path_h索引是patch的行
        for index_y in path_w:  # path_w索引是patch的列
            patch = fill_arr[index_x - patch_size: index_x, index_y - patch_size: index_y]
            cut_patches.append(patch)
    return cut_patches, number_h, number_w, arr_h, arr_w

def combine(patches, patch_size, number_h, number_w, block_h, block_w):
    """
    完整数据用get_patches切分后，将数据进行还原会原始数据块大小
    :param patches: get_patches切分后的结果，以列表形式传入
    :param patch_size: 数据切片patch的大小
    :param number_h: 高方向切出的patch数量
    :param number_w: 宽方向切出的patch数量
    :param block_h: 地震数据块的高
    :param block_w: 地震数据块的宽
    :return: 还原后的地震数据块
    """
    # 将列表patch1中的数据取出，转换成二维矩阵。按照列表元素顺序拼接。
    # patch_size = int(patch_size)
    temp = np.zeros((int(patch_size), 1), dtype=np.float32)  # 临时拼接矩阵，后面要删除
    # 取出patch1中的每一个元素，在列方向(axis=1)拼接
    for i in range(len(patches)):
        temp = np.append(temp, patches[i], axis=1)
    # 删除temp后，此时temp1的维度是 patch_size * patch_size*number_h*number_w
    temp1 = np.delete(temp, 0, axis=1)  # 将temp删除

    # 将数据变成 (patch_size*number_h) * (patch_size*number_w)
    test = np.zeros((1, int(patch_size*number_w)), dtype=np.float32)  # 临时拼接矩阵，后面要删除
    # 让temp1每隔patch_size/2*number_w列就进行一个换行操作
    for j in range(0, int(patch_size*number_h*number_w), int(patch_size*number_w)):
        test = np.append(test, temp1[:, j:j + int(patch_size*number_w)], axis=0)
    test1 = np.delete(test, 0, axis=0)  # 将test删除
    block_data = test1[0:block_h, 0:block_w]
    return block_data
```


The README should include the following sections:

1. Project Title
2. Brief description (1-2 sentences)
3. Features
4. Installation instructions
5. Usage examples
6. Configuration options (if applicable) 
7. Contribution guidelines
8. Testing instructions
9. License
10. Acknowledgements/Credits

Write the content in Markdown format. Use your analysis of the code to generate accurate and helpful content, but also explain things clearly for users who may not be familiar with the implementation details.

Feel free to infer reasonable details if needed, but try to stick to what can be determined from the codebase itself. Let me know if you have any other questions as you're writing!