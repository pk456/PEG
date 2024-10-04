import logging
import pickle

import torch
import tqdm
from torch import nn, optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model.paper import Paper
from model.qb import QB
from model.reward import Reward

logging.getLogger().setLevel(logging.INFO)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, condition_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.3),
            nn.Tanh()
        )

    def forward(self, z, condition):
        z = torch.cat((z, condition), dim=-1)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, condition_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.3),
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        x = torch.cat((x, condition), dim=-1)
        return self.net(x)


class ExamDataset(Dataset):
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c = torch.tensor(np.array(self.data[idx][0]), dtype=torch.float32).view(-1)
        qb = torch.tensor(self.data[idx][1], dtype=torch.int)
        students_concept_status = torch.tensor(self.data[idx][2], dtype=torch.float32)
        return c, qb, students_concept_status


class ExamGAN(object):
    def __init__(self, args):
        # 参数设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_qb = args.all_num_questions
        self.condition_dim = args.num_concepts * 2
        self.random_dim = args.random_dim
        self.final_dim = 1

        # 定义模型
        self.generator = Generator(input_dim=self.random_dim, condition_dim=self.condition_dim, hidden_dim=256,
                                   output_dim=self.num_qb).to(self.device)
        self.discriminator = Discriminator(input_dim=self.num_qb, condition_dim=self.condition_dim, hidden_dim=256,
                                           output_dim=self.final_dim).to(self.device)

    def train(self, train_data, batch_size, val=True, epoch=200, lr=0.001):
        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        real_labels = torch.ones(batch_size, self.final_dim).to(self.device)
        fake_labels = torch.zeros(batch_size, self.final_dim).to(self.device)

        # 训练模型
        self.generator.train()
        for e in range(epoch):
            # 统计损失
            d_loss_mean = []
            g_loss_mean = []

            for c, qb, _ in tqdm.tqdm(train_data, desc="Epoch %s" % e):
                # 将张量放在device下
                c = c.to(self.device)
                qb = qb.to(self.device)
                noise = torch.randn(batch_size, self.random_dim).to(self.device)

                # 训练判别器
                optimizer_d.zero_grad()
                real_output = self.discriminator(qb, c)
                real_loss = criterion(real_output, real_labels)

                fake_qb = self.generator(noise, c)
                fake_output = self.discriminator(fake_qb.detach(), c)
                fake_loss = criterion(fake_output, fake_labels)

                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_d.step()

                # 训练生成器
                optimizer_g.zero_grad()
                fake_output = self.discriminator(fake_qb, c)
                g_loss = criterion(fake_output, real_labels)
                g_loss.backward()
                optimizer_g.step()

                d_loss_mean.append(d_loss.item())
                g_loss_mean.append(g_loss.item())
            # 输出损失
            logging.info('d_loss: {:.4f}, g_loss: {:.4f}'.format(np.mean(d_loss_mean), np.mean(g_loss_mean)))
            # # 在验证集上评估模型
            # if val:

    def valuate(self, val_data, batch_size, reward):
        # 在验证集上评估模型
        self.generator.eval()
        with self.generator.no_grad():
            for c, qb in tqdm.tqdm(val_data):
                noise = torch.randn(batch_size, self.random_dim).to(self.device)
                fake_qb = self.generator(noise, c)
                # 计算评估指标
                pass

    def test(self, test_loader):
        # 在测试集上评估模型
        pass

    def generate_exam_script(self, c):
        # 生成试卷脚本
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(self.random_dim).to(self.device)
            fake_qb = self.generator(noise, c)
            # 将生成的试卷脚本转换为文本
            return fake_qb

    def save_model(self, path):
        # 保存模型
        with open(path + '_generator', 'wb') as f:
            torch.save(self.generator.state_dict(), f)
        with open(path + '_discriminator', 'wb') as f:
            torch.save(self.discriminator.state_dict(), f)

    def load_model(self, path):
        # 加载模型
        with open(path + '_generator', 'rb') as f:
            self.generator.load_state_dict(torch.load(f))
        with open(path + '_discriminator', 'rb') as f:
            self.discriminator.load_state_dict(torch.load(f))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ExamDataset('../data/c_filter/gan/train_data.pkl')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    exam_gan = ExamGAN()
    # exam_gan.train(train_data=train_loader, batch_size=32)
    # exam_gan.save_model('../data/c_filter/gan/exam_gan')
    exam_gan.load_model('../data/c_filter/gan/exam_gan')
    val_dataset = ExamDataset('../data/c_filter/gan/val_data.pkl')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    qb = QB()
    for c, qb, student_concept_status in val_loader:
        c = c.to(device)
        qb = qb.to(device)

        fake_qb = exam_gan.generate_exam_script(c)
        print(fake_qb)
        questions = torch.topk(fake_qb[0], k=10)

        paper = Paper(questions, )
    pass
