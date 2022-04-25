from torch.nn import *
from torch.utils.tensorboard import SummaryWriter
import torch


class MyNetwork(Module):
    def __init__(self) -> None:
        super().__init__()

        # 网络结构
        self.module = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(4096, 64),
            ReLU(),
            Linear(64, 2)
        )

    # 前向传播函数
    def forward(self, x):
        x = self.module(x)
        return x

if __name__=="__main__":
    model = MyNetwork()
    input = torch.rand([1, 3, 64,64])
    writer = SummaryWriter(".\\MyNetworkStruct")
    writer.add_graph(model=model, input_to_model=input)
    writer.close()