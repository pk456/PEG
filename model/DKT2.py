import logging
import os.path
import torch
import tqdm
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score


class Net(nn.Module):
    def __init__(self, num_concepts, hidden_size, num_layers, device):
        super(Net, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = num_layers
        self.lstm = nn.LSTM(num_concepts * 2, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, num_concepts)
        self.device = device

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        res = torch.sigmoid(self.fc(out))
        return res


def process_raw_pred(raw_question_matrix, raw_pred, num_concepts: int) -> tuple:

    non_zero_indices = torch.nonzero(raw_question_matrix)
    unique_first_column, inverse_indices = torch.unique(non_zero_indices[:, 0], return_inverse=True)
    unique_rows_tensor = torch.stack([non_zero_indices[inverse_indices == i][0] for i in unique_first_column])
    truth = unique_rows_tensor[1:,1]// num_concepts

    length = len(truth)

    raw_question = raw_question_matrix[1:].reshape(-1,122)
    columns_to_keep = ~torch.all(raw_question == 0, dim=1)
    raw_question = raw_question[columns_to_keep]

    knowledge_status = raw_pred[: length]

    pred = knowledge_status * raw_question

    mask = torch.ne(pred, 0)
    knowledge_match = torch.where(mask, pred, 1)
    students_q_score = torch.prod(knowledge_match, dim=-1)

    # pred = pred.gather(1, records.view(-1, 1)).flatten()
    # truth = torch.nonzero(raw_question_matrix)[1:, 1] // num_concepts
    # truth = torch.nonzero(raw_question_matrix) // torch.tensor([1, num_concepts]).to(raw_question_matrix.device)
    return students_q_score, truth


class DKT(object):
    def __init__(self, num_concepts, hidden_size, num_layers, device=None):
        super(DKT, self).__init__()
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.num_concepts = num_concepts
        self.dkt_model = Net(num_concepts, hidden_size, num_layers, device).to(device)

    def train(self, train_data, test_data=None, save_model_file=None, *, epoch: int, lr=0.002) -> ...:
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr)

        for e in range(epoch):
            self.dkt_model.train()
            all_pred, all_target = torch.Tensor([]).to(self.device), torch.Tensor([]).to(self.device)
            for batch in tqdm.tqdm(train_data, "Epoch %s" % e):
                batch = batch.to(self.device)
                integrated_pred = self.dkt_model(batch)
                batch_size = batch.shape[0]
                for student in range(batch_size):
                    pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_concepts)
                    all_pred = torch.cat([all_pred, pred])
                    all_target = torch.cat([all_target, truth.float()])

            loss = loss_function(all_pred, all_target)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("[Epoch %d] LogisticLoss: %.6f" % (e, loss))

            if test_data is not None:
                auc = self.eval(test_data)
                print("[Epoch %d] auc: %.6f" % (e, auc))
            if save_model_file is not None:
                directory = os.path.dirname(save_model_file)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.save(save_model_file + str(e))

    def eval(self, test_data) -> float:
        self.dkt_model.eval()
        y_pred = torch.Tensor([]).to(self.device)
        y_truth = torch.Tensor([]).to(self.device)
        for batch in tqdm.tqdm(test_data, "evaluating"):
            batch = batch.to(self.device)
            integrated_pred = self.dkt_model(batch)
            batch_size = batch.shape[0]
            for student in range(batch_size):
                pred, truth = process_raw_pred(batch[student], integrated_pred[student], self.num_concepts)
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])

        return roc_auc_score(y_truth.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

    def save(self, filepath):
        torch.save(self.dkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
