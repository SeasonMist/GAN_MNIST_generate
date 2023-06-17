import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils import data
import os
import glob
from PIL import Image

import time


class AiGcMn:
    # 独热编码
    # 输入x代表默认的torchvision返回的类比值，class_count类别值为10
    def one_hot(self, x):
        return torch.eye(10)[x, :]  # 切片选取，第一维选取第x个，第二维全要

    #=================================================================================================================================#

    # 定义生成器
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 128 * 7 * 7)
            self.bn1 = nn.BatchNorm1d(128 * 7 * 7)
            self.linear2 = nn.Linear(100, 128 * 7 * 7)
            self.bn2 = nn.BatchNorm1d(128 * 7 * 7)
            self.deconv1 = nn.ConvTranspose2d(256, 128,
                                              kernel_size=(3, 3),
                                              padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.deconv2 = nn.ConvTranspose2d(128, 64,
                                              kernel_size=(4, 4),
                                              stride=2,
                                              padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.deconv3 = nn.ConvTranspose2d(64, 1,
                                              kernel_size=(4, 4),
                                              stride=2,
                                              padding=1)

        def forward(self, x1, x2):
            x1 = F.relu(self.linear1(x1))
            x1 = self.bn1(x1)
            x1 = x1.view(-1, 128, 7, 7)
            x2 = F.relu(self.linear2(x2))
            x2 = self.bn2(x2)
            x2 = x2.view(-1, 128, 7, 7)
            x = torch.cat([x1, x2], axis=1)
            x = F.relu(self.deconv1(x))
            x = self.bn3(x)
            x = F.relu(self.deconv2(x))
            x = self.bn4(x)
            x = torch.tanh(self.deconv3(x))
            return x

    #=================================================================================================================================#

    # 定义判别器
    # input:1，28，28的图片以及长度为10的condition
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1*28*28)
            self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
            self.bn = nn.BatchNorm2d(128)
            self.fc = nn.Linear(128*6*6, 1) # 输出一个概率值

        def forward(self, x1, x2):
            x1 =F.leaky_relu(self.linear(x1))
            x1 = x1.view(-1, 1, 28, 28)
            x = torch.cat([x1, x2], axis=1)
            x = F.dropout2d(F.leaky_relu(self.conv1(x)))
            x = F.dropout2d(F.leaky_relu(self.conv2(x)))
            x = self.bn(x)
            x = x.view(-1, 128*6*6)
            #x = torch.sigmoid(self.fc(x))
            x = self.fc(x)
            return x

    #=================================================================================================================================#

    def __init__(self, n, input_label):
        transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(0.5, 0.5)])

        dataset = torchvision.datasets.MNIST('data',
                                             train=True,
                                             transform=transform,
                                             target_transform=self.one_hot,
                                             download=True)
        self.dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)

        # 初始化模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gen = self.Generator().to(self.device)
        self.dis = self.Discriminator().to(self.device)

        # 损失计算函数
        self.loss_function = torch.nn.BCELoss()

        # 定义优化器
        self.d_optim = torch.optim.RMSprop(self.dis.parameters(), lr=5e-5)
        self.g_optim = torch.optim.RMSprop(self.gen.parameters(), lr=5e-5)
        
        # 定义初始种子
        self.noise_seed = torch.randn(n, 100, device=self.device)

        self.label_seed = input_label
        print(self.label_seed)
        self.label_seed_onehot = self.one_hot(self.label_seed).to(self.device)
        #print(self.label_seed_onehot)

    #=================================================================================================================================#

    # 定义可视化函数
    def generate_and_save_images(self, model, epoch, label_input, noise_input):
        predictions = np.squeeze(model(label_input, noise_input).cpu().numpy())
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((predictions[i] + 1) / 2, cmap='gray')
            plt.axis("off")
        plt.show()
        return predictions

    #=================================================================================================================================#

    def generate(self, n, epoch_times):
        results = np.ndarray(shape=(0,))
        
        # 开始训练
        D_loss = []
        G_loss = []

        weight_cliping_limit = 0.01

        one = torch.FloatTensor([1])
        mone = one * -1
        if self.device == 'cuda':
            one = one.cuda(0)
            mone = mone.cuda(0)

        t0 = time.time() # 初始时间

        # 训练循环
        for epoch in range(epoch_times):
            # 计时 #
            t = time.time()
            # 计时 #

            d_epoch_loss = 0
            g_epoch_loss = 0
            count = len(self.dataloader.dataset)
            # 对全部的数据集做一次迭代
            for step, (img, label) in enumerate(self.dataloader):
                img = img.to(self.device)
                label = label.to(self.device)
                size = img.shape[0]
                random_noise = torch.randn(size, 100, device=self.device)

                self.d_optim.zero_grad()

                for p in self.dis.parameters():
                    p.requires_grad = True

                for d_iter in range(5):
                    self.d_optim.zero_grad()

                    for p in self.dis.parameters():
                        p.data.clamp_(-weight_cliping_limit, weight_cliping_limit)

                    real_output = self.dis(label, img)
                    d_real_loss = real_output.mean(0).view(1)
                    d_real_loss.backward(one) #求解梯度

                    # 得到判别器在生成图像上的损失
                    gen_img = self.gen(label,random_noise)
                    fake_output = self.dis(label, gen_img.detach())  # 判别器输入生成的图片，f_o是对生成图片的预测结果
                    d_fake_loss = fake_output.mean(0).view(1)
                    d_fake_loss.backward(mone)

                    d_loss = - d_real_loss + d_fake_loss
                    self.d_optim.step()  # 优化
                for p in self.dis.parameters():
                    p.requires_grad = False

                # 得到生成器的损失
                self.g_optim.zero_grad()
                fake_output = self.dis(label, gen_img)
                g_loss = fake_output.mean().mean(0).view(1)
                g_loss.backward(one)
                self.g_optim.step()

                with torch.no_grad():
                    d_epoch_loss += d_loss.item()
                    g_epoch_loss += g_loss.item()
            with torch.no_grad():
                d_epoch_loss /= count
                g_epoch_loss /= count
                D_loss.append(d_epoch_loss)
                G_loss.append(g_epoch_loss)
                if epoch % 1 == 0:
                    print('Epoch:', epoch)
                    results = self.generate_and_save_images(self.gen, epoch, self.label_seed_onehot, self.noise_seed)

            # 计时 #
            print(f'第{epoch}次迭代用时:{time.time() - t:.4f}s')
            print('Discriminator Loss:', d_epoch_loss, 'Generator Loss:', g_epoch_loss)
            # 计时 #

        print(f'总计用时:{time.time() - t0:.4f}s')

        imgs = torch.from_numpy(np.zeros((n, 1, 28, 28)))
        results_tensor = torch.from_numpy(results)
        for i in range(n):
            imgs[i,0] = results_tensor[i]
        return imgs
    

# 调用实例
input_label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
n = input_label.size()[0]
epoch_times = 25

aigcmn = AiGcMn(n, input_label)
res = aigcmn.generate(n, epoch_times)