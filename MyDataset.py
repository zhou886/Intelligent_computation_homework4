import torch
from torch.utils.data import Dataset
import cv2 as cv
import os
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, root_path, train, size, k_fold: int = 0, manification: int = 1, rand_crop: bool = False):
        super().__init__()
        self.root_path = root_path                      # 保存根目录
        self.label_path = os.listdir(self.root_path)    # 获得所有标签
        self.imgAndLabel = []                           # 用于存放图像地址和对应标签
        # 是否使用随机裁切
        if rand_crop:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.RandomCrop(256, 256), transforms.Resize((64, 64))])
        else:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((64, 64))])
        size //= 2
        if k_fold == 0:
            # 不使用K折交叉验证
            if train:
                # 如果用作训练集
                for label in self.label_path:
                    path = os.path.join(self.root_path, label)
                    img_path = os.listdir(path)

                    # 将数据集的前80%放在imgAndLabel中用作训练集
                    total_img_size = round(size*0.8)
                    for i in range(total_img_size):
                        if rand_crop:
                            for j in range(manification):
                                self.imgAndLabel.append(
                                    (os.path.join(self.root_path,
                                                  label, img_path[i]), label)
                                )
            else:
                # 如果用作测试集
                for label in self.label_path:
                    path = os.path.join(self.root_path, label)
                    img_path = os.listdir(path)

                    # 将数据集的后20%放在imgAndLabel中用作测试集
                    total_img_size = round(size*0.8)
                    for i in range(total_img_size, size):
                        if rand_crop:
                            for j in range(manification):
                                self.imgAndLabel.append(
                                    (os.path.join(self.root_path,
                                                  label, img_path[i]), label)
                                )
        else:
            # 使用K折交叉验证
            self.k = k_fold
            for label in self.label_path:
                path = os.path.join(self.root_path, label)
                img_path = os.listdir(path)
                for i in range(size):
                    if rand_crop:
                        for j in range(manification):
                            self.imgAndLabel.append(
                                (os.path.join(self.root_path, label, img_path[i]), label))
                                
        self.size = len(self)
        self.cache = {}

    def get_k_fold_data(self, i):
        """
        获取以第i折作为测试集，其他作为训练集的数据集
        """
        if 0 <= i < self.k:
            train_set = []
            test_set = []
            fold_size = self.size // self.k
            for j in range(self.k):
                for k in range(j*fold_size, (j+1)*fold_size):
                    if i == j:
                        test_set.append(self[k])
                    else:
                        train_set.append(self[k])
            return train_set, test_set
        else:
            raise Exception()

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        # 获得图像路径和标签
        img_path, label = self.imgAndLabel[index]

        # 读取图像
        img = cv.imread(img_path)

        # 如果有变换就对图像应用变换
        if self.transform:
            img = self.transform(img)

        # 将标签转换为对应的tensor类型的数据方便使用
        label_tensor = 0
        for i in range(len(self.label_path)):
            if label == self.label_path[i]:
                label_tensor = torch.tensor(i)
                break

        self.cache[index] = (img, label_tensor)
        return  self.cache[index]

    def __len__(self):
        return len(self.imgAndLabel)


# 单元测试
if __name__ == "__main__":
    myDataset = MyDataset(r'.\dataset', train=False, transform=None)
    for data in myDataset:
        img, label = data
        print(img, '\n', label, '\n')
