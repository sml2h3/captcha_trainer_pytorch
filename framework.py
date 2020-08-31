#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2020/8/26 22:44
@Author:   sml2h3
@File:     framework
@Software: PyCharm
"""

from torch import optim
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import *
from utils.exception import *


class FrameWork(nn.Module):

    def __init__(self, config):
        super(FrameWork, self).__init__()
        self.inputs = None
        self.labels = None
        self.config = config
        if torch.cuda.is_available() and self.config['System']['GPU'] and int(self.config['System']['GPU_ID']) > -1:
            self.device = torch.device('cuda:{}'.format(self.config['System']['GPU_ID']))
        else:
            self.device = torch.device('cpu')
        self.image_channel = self.config['Model']['ImageChannel']
        self.resize = self.config['Model']['RESIZE']
        self.cnn_type = self.config['Train']['CNN']['NAME']
        self.paramters = []
        if self.cnn_type == CNN.MobileNetV2.value:
            from torchvision.models import mobilenet_v2
            self.cnn = mobilenet_v2().features
            self.cnn[0][0] = torch.nn.Conv2d(int(self.image_channel), 32, (3, 3), stride=(2, 2), padding=(1, 1),
                                             bias=False)
            self.cnn.to(device=self.device)
            self.paramters.append({'params': self.cnn.parameters()})
            self.out_size = 1280
        elif self.cnn_type == CNN.EfficientNetb0.value:
            from efficientnet_pytorch import EfficientNet
            self.cnn = EfficientNet.from_name('efficientnet-b0')
            self.cnn._conv_stem = torch.nn.Conv2d(int(self.image_channel), 32, kernel_size=3, stride=2, bias=False)
            self.cnn.to(device=self.device)
            self.paramters.append({'params': self.cnn.parameters()})
            self.out_size = 1280
        else:
            raise CnnNotFoundError("CNN Name not found!")

        rnn = self.config['Train']['RNN']['NAME']
        self.hidden_num = int(self.config['Train']['LSTM']['HIDDEN_NUM'])
        dropout = int(self.config['Train']['LSTM']['DROPOUT'])
        if rnn == RNN.LSTM.value:
            self.lstm = nn.LSTM(input_size=self.out_size, hidden_size=self.hidden_num, num_layers=2, bidirectional=True,
                                dropout=dropout)
            self.lstm.to(device=self.device)
            self.paramters.append({'params': self.lstm.parameters()})
        else:
            raise RnnNotFoundError("RNN Name not found!")

        self.charset = self.config['Model']['CharSet']
        self.charset = json.loads(self.charset)
        self.charset_len = len(self.charset)

        self.fc = nn.Linear(in_features=self.hidden_num * 2, out_features=self.charset_len)
        self.fc.to(device=self.device)
        self.paramters.append({'params': self.fc.parameters()})

        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
        self.ctc_loss.to(device=self.device)
        self.paramters.append({'params': self.ctc_loss.parameters()})

        optimizer = self.config['Train']['OPTIMIZER']
        self.lr = self.config['Train']['LR']
        if optimizer == OPTIMIZER.Momentum.value:
            self.optimizer = optim.SGD(self.paramters, lr=self.lr, momentum=0.9)
        elif optimizer == OPTIMIZER.Adma.value:
            self.optimizer = optim.Adam(self.paramters, lr=self.lr, betas=(0.9, 0.99))
        else:
            raise OptimizerNotFoundError("Optimizer Name not found!")

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

    def forward(self, inputs):
        self.cnn.train()
        self.lstm.train()
        self.fc.train()
        predict = self.get_feature(inputs)
        outputs = predict.max(2)[1].transpose(0, 1)
        return outputs

    def train_op(self, inputs, labels, label_length):
        self.cnn.train()
        self.lstm.train()
        self.fc.train()
        self.ctc_loss.train()
        predict = self.get_feature(inputs)
        loss, lr = self.get_loss(predict, labels, label_length)
        return loss, lr

    def test_op(self, inputs, labels, label_length):
        self.cnn.eval()
        self.lstm.eval()
        self.fc.eval()
        self.ctc_loss.eval()
        predict = self.get_feature(inputs)
        outputs = predict.max(2)[1].transpose(0, 1)
        pred_decode_labels = []
        for pred_labels in outputs:
            decoded = []
            for item in pred_labels:
                if item != 0:
                    decoded.append(item.item())
            pred_decode_labels.append(decoded)
        labels_list = []
        labels = labels.tolist()
        i = 0
        for idx in label_length.tolist():
            labels_list.append(labels[i: i + idx])
            i += idx
        if len(labels_list) != len(pred_decode_labels):
            raise PredictLabelLengthIsNotMatch("origin labels length is {}, but pred labels length is {}".format(
                len(labels_list, len(pred_decode_labels))))
        correct_list = []
        error_list = []
        for ids in range(len(labels_list)):
            if labels_list[ids] == pred_decode_labels[ids]:
                correct_list.append(ids)
            else:
                error_list.append(ids)

        return pred_decode_labels, labels_list, correct_list, error_list

    def get_feature(self, inputs):
        inputs = Variable(inputs).to(device=self.device)
        if self.cnn_type == CNN.EfficientNetb0.value:
            x = self.cnn.extract_features(inputs)
        else:
            x = self.cnn(inputs)
        x = torch.reshape(x, (x.shape[0], x.shape[2] * x.shape[3], x.shape[1])).to(device=self.device)
        x, _ = self.lstm(x)
        x = torch.reshape(x, (-1, self.hidden_num * 2)).to(device=self.device)
        x = self.fc(x)
        x = torch.reshape(x, [inputs.shape[0], -1, self.charset_len]).to(device=self.device)
        predict = torch.transpose(x, 1, 0).to(device=self.device)
        return predict

    def get_loss(self, pred, labels, label_length):
        labels = Variable(labels).to(device=self.device)
        log_pred = pred.log_softmax(2)
        seq_len = torch.IntTensor([log_pred.shape[0]] * log_pred.shape[1]).to(device=self.device)
        loss = self.ctc_loss(log_pred.cpu(), labels, seq_len, label_length)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        return loss.item(), self.scheduler.state_dict()['_last_lr'][-1]

    def save_model(self, PATH, net):
        torch.save(net.state_dict(), PATH)
