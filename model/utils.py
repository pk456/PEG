import torch
import torch.nn.functional as F


# 知识点覆盖率
def dis(E, W):
    return torch.norm(E - W)


def dif(R, avg):
    avg_R = torch.mean(R)
    return torch.abs(avg_R - avg)


def div(R, Z):
    log_Z = torch.log(Z)
    # todo:这里为什么要log_Z，然后反过来
    kl_div_result = F.kl_div(log_Z, R, reduction='sum')
    return kl_div_result




if __name__ == '__main__':
    R = torch.tensor([1.0, 2.0, 3.0])
    Z = torch.tensor([0.1, 0.2, 0.3])
    print(div(R, Z))
    print(torch.sum(R * (torch.log(R) - torch.log(Z))))
