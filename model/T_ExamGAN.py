import os

import numpy as np
import torch
import tqdm
from torch import nn

from model.ExamGAN import Generator, Discriminator


class T_ExamGAN(object):
    def __init__(self, args):
        # 参数设置
        self.device = torch.device(('cuda:%d' % args.gpu) if torch.cuda.is_available() else "cpu")
        self.num_qb = args.all_num_questions
        self.condition_dim = args.num_concepts * 2
        self.random_dim = args.random_dim
        self.final_dim = 1
        self.num_questions = args.num_questions

        # 定义模型
        self.generator_a = Generator(input_dim=self.random_dim, condition_dim=self.condition_dim, hidden_dim=256,
                                     output_dim=self.num_qb).to(self.device)
        self.generator_b = Generator(input_dim=self.random_dim, condition_dim=self.condition_dim, hidden_dim=256,
                                     output_dim=self.num_qb).to(self.device)
        self.discriminator = Discriminator(input_dim=self.num_qb, condition_dim=self.condition_dim, hidden_dim=256,
                                           output_dim=self.final_dim).to(self.device)

    # 算法2
    def train(self, train_data, batch_size, val=True, epoch=200, lr=0.001):
        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        criterion2 = JaccardLoss(1e-6)
        optimizer_ga = torch.optim.Adam(self.generator_a.parameters(), lr=lr)
        optimizer_gb = torch.optim.Adam(self.generator_a.parameters(), lr=lr)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        real_labels = torch.ones(batch_size, self.final_dim).to(self.device)
        fake_labels = torch.zeros(batch_size, self.final_dim).to(self.device)

        # 训练模型
        self.generator_a.train()
        self.generator_b.train()
        for e in range(epoch):
            # 统计损失
            d_loss_mean = []
            g_loss_a_mean = []
            g_loss_b_mean = []

            for c, qb, _ in tqdm.tqdm(train_data, desc="Epoch %s" % e):
                # 将张量放在device下
                c = c.to(self.device)
                qb = qb.to(self.device)
                noise = torch.randn(batch_size, self.random_dim).to(self.device)

                # 训练判别器
                optimizer_d.zero_grad()
                real_output = self.discriminator(qb, c)
                real_loss = criterion(real_output, real_labels)

                fake_qb_a = self.generator_a(noise, c)
                fake_output_a = self.discriminator(fake_qb_a.detach(), c)
                fake_loss_a = criterion(fake_output_a, fake_labels)

                fake_qb_b = self.generator_b(noise, c)
                fake_output_b = self.discriminator(fake_qb_b.detach(), c)
                fake_loss_b = criterion(fake_output_b, fake_labels)

                d_loss = real_loss + fake_loss_a + fake_loss_b
                d_loss.backward()
                optimizer_d.step()

                # 训练生成器A
                optimizer_ga.zero_grad()
                fake_output_a = self.discriminator(fake_qb_a, c)
                g_loss_a = criterion(fake_output_a, real_labels)
                g_loss_a.backward()
                optimizer_ga.step()

                # 训练生成器B
                optimizer_gb.zero_grad()
                fake_output_b = self.discriminator(fake_qb_b, c)
                g_loss_b = criterion(fake_output_b, real_labels)
                g_loss_b.backward()
                optimizer_gb.step()

                d_loss_mean.append(d_loss.item())
                g_loss_a_mean.append(g_loss_a.item())
                g_loss_b_mean.append(g_loss_b.item())

                # 进行重复率判断
                optimizer_ga.zero_grad()
                optimizer_gb.zero_grad()
                fake_qb_a = self.generator_a(noise, c)
                fake_qb_b = self.generator_b(noise, c)
                ga_mask = self.generate_new_qb(fake_qb_a)
                gb_mask = self.generate_new_qb(fake_qb_b)
                g_loss_ab = criterion2(fake_qb_a, fake_qb_b, ga_mask, gb_mask)
                if g_loss_ab.item() > 0.3:
                    # todo：自定义的这个损失函数应该如何利用，backward具体含义，注：criterion2中是新创建的两个tensor
                    g_loss_ab.backward()
                    # todo:是应该每轮换一个生成器，还是应该每次换一个
                    if e % 2 == 0:
                        optimizer_ga.step()
                    else:
                        optimizer_gb.step()

            # 输出损失
            print("Epoch [{}/{}], d_loss: {:.4f}, g_loss_a: {:.4f}, g_loss_b: {:.4f}".format(e, epoch,
                                                                                             np.mean(d_loss_mean),
                                                                                             np.mean(g_loss_a_mean),
                                                                                             np.mean(g_loss_b_mean)))

    def generate_exam_script(self, c):
        # 生成试卷脚本
        self.generator_a.eval()
        with torch.no_grad():
            noise = torch.randn(self.random_dim).to(self.device)
            fake_qb_a = self.generator_a(noise, c)
            fake_qb_b = self.generator_b(noise, c)
            # 将生成的试卷脚本转换为文本
            return torch.stack([fake_qb_a, fake_qb_b])

    # 挑选概率最高的100个组成试卷
    def generate_new_qb(self, fake_qb):
        _, questions = torch.topk(fake_qb, k=self.num_questions, dim=-1)
        new_qb = torch.zeros_like(fake_qb)

        new_qb.scatter_(-1, questions, 1)
        return new_qb

    def save_model(self, path):
        # 保存模型
        with open(path + '_t_generator_a', 'wb') as f:
            torch.save(self.generator_a.state_dict(), f)
        with open(path + '_t_generator_b', 'wb') as f:
            torch.save(self.generator_b.state_dict(), f)
        with open(path + '_t_discriminator', 'wb') as f:
            torch.save(self.discriminator.state_dict(), f)

    def load_model(self, path):
        # 加载模型
        with open(path + '_t_generator_a', 'rb') as f:
            self.generator_a.load_state_dict(torch.load(f))
        with open(path + '_t_generator_b', 'rb') as f:
            self.generator_b.load_state_dict(torch.load(f))
        with open(path + '_t_discriminator', 'rb') as f:
            self.discriminator.load_state_dict(torch.load(f))

    # 判断path是否保存了model
    def exist_model(self, path):
        return (os.path.exists(path + '_t_generator_a') and os.path.exists(path + '_t_generator_b')
                and os.path.exists(path + '_t_discriminator'))


class JaccardLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, ga, gb, ga_mask, gb_mask):
        ga = ga * ga_mask
        gb = gb * gb_mask

        # 计算交集和并集
        intersection = (ga * gb).sum(dim=-1)
        union = (ga + gb).sum(dim=-1) - intersection

        # 计算 Jaccard 指数
        jaccard_index = (intersection + self.smooth) / (union + self.smooth)

        # 计算 Jaccard 损失
        jaccard_loss = jaccard_index.mean()

        return jaccard_loss
