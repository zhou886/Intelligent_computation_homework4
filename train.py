from torchvision.transforms.transforms import Resize, ToTensor
from MyDataset import MyDataset
from MyNetwork import MyNetwork
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import os
from time import strftime
from datetime import datetime
from multiprocessing import Process

def main():
    learning_rate = 0.001
    batch_size = 128

    # 创建训练集和测试集
    test_set = MyDataset(r'.\dataset', train=False, size=set_size)
    train_set = MyDataset(r'.\dataset', train=True, size=set_size)

    for set_size in (512,1024,2048,4096,8192):
        process_list = []
        for i in range(5):
            p = Process(target=train, args=(test_set, train_set, set_size, learning_rate, batch_size))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()


def train(test_set, train_set, set_size, learning_rate=0.001, batch_size=256):
    # 创建神经网络
    myNetwork = MyNetwork()

    # 创建损失函数，使用交叉熵
    loss_function = nn.CrossEntropyLoss()

    # 如果可以使用cuda加速的话就把神经网络和损失函数转换为cuda形式
    if torch.cuda.is_available():
        myNetwork = myNetwork.cuda()
        loss_function = loss_function.cuda()

    # 创建训练集和测试集的加载器Dataloader
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # 使用随机梯度下降算法
    optimizer = torch.optim.SGD(myNetwork.parameters(), lr=learning_rate)

    # 创建存放训练模型的文件夹
    os.mkdir(".\\model_setsize{}_SGD_lr{}_batchsize{}".format(set_size, learning_rate, batch_size))

    # 初始化总训练次数和总测试次数
    total_train_step = 0
    total_test_step = 0

    # 设置epoch
    epoch = 100

    # 使用tensoboard查看训练变化过程
    writer = SummaryWriter(
        "logs_setsize{}_SGD_lr{}_batchsize{}".format(set_size,learning_rate, batch_size))

    for i in range(epoch):
        print("------Epoch {} start------".format(i))

        # 使用训练模式，使随机失活和正则化层正常工作
        myNetwork.train()

        # 初始化总训练损失和总训练准确度
        total_train_loss = 0
        total_train_accuracy = 0

        for data in train_dataloader:  # 从dataloader中提出数据
            imgs, targets = data  # 分离图片和标签

            # 如果可以使用cuda加速的话就把图片和标签转换为cuda形式
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            # 网络的训练过程
            output = myNetwork(imgs)                        # 获得网络结果
            loss = loss_function(output, targets)           # 获得本次训练的损失
            optimizer.zero_grad()                           # 清除上次训练产生的梯度
            loss.backward()                                 # 反向传播损失，使用计算图计算各参数梯度
            optimizer.step()                                # 使用优化器优化网络参数

            accuracy = (output.argmax(1) == targets).sum()  # 获得本次训练的准确度
            total_train_accuracy += accuracy                # 累加本次训练的准确度
            total_train_loss += loss.item()                 # 累加本次训练的损失

        # 使用测试模式，关闭随机失活和正则化层
        myNetwork.eval()

        # 初始化总测试损失和总测试准确度
        total_test_loss = 0
        total_test_accuracy = 0

        with torch.no_grad():  # 不用使用梯度
            for data in test_dataloader:
                imgs, targets = data
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    targets = targets.cuda()

                output = myNetwork(imgs)
                loss = loss_function(output, targets)
                total_test_loss += loss.item()
                accuracy = (output.argmax(1) == targets).sum()
                total_test_accuracy += accuracy

        # 计算训练集和测试集上的准确度
        total_train_accuracy = 1.0*total_train_accuracy/train_set.size
        total_test_accuracy = 1.0*total_test_accuracy/test_set.size

        # 把本轮训练的损失、准确度和本轮测试的损失、准确度写入tensorboard之中
        writer.add_scalar('train_loss', total_train_loss, total_train_step)
        writer.add_scalar('train_accuracy',
                          total_train_accuracy, total_train_step)
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar(
            "test_accuracy", total_test_accuracy, total_test_step)

        # 训练、测试步长加一
        total_train_step += 1
        total_test_step += 1

        # 保存模型
        now = datetime.now()
        torch.save(myNetwork.state_dict(),
                   os.path.join(os.getcwd(), 'model_setsize{}_SGD_lr{}_batchsize{}'.format(set_size, learning_rate, batch_size), "MyNetwork_time{}_accuracy{}.pth".format(
                       now.strftime(r"%Y-%m%d-%H%M-%S"), round(total_test_accuracy.item(), 2)))
                   )
        print("Network module has been saved.")

    writer.close()
