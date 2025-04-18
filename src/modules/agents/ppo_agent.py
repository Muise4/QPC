import torch
import torch.nn as nn
import torch.nn.functional as F
    
class PPO_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(PPO_Agent, self).__init__()
        self.args = args
        self.conv = nn.Sequential(
        nn.Conv2d(in_channels=input_shape[0], out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.Tanh(),
        nn.Flatten()
        )
        self.fc1 = nn.Linear(args.out_channels * (input_shape[1] - 2) * (input_shape[2] - 2), args.rnn_hidden_dim)
        # self.lstm = nn.LSTM(
        #     input_size=args.rnn_hidden_dim,
        #     hidden_size=args.rnn_hidden_dim,
        #     batch_first=True
        # )
        self.lstm = nn.LSTMCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, inputs, h, c):
        inputs = inputs / 255.0
        # 输入绝大多数都是0呀！
        x = self.conv(inputs)
        x = F.relu(self.fc1(x))
        h, c = self.lstm(x, (h.reshape(-1, self.args.rnn_hidden_dim), c.reshape(-1, self.args.rnn_hidden_dim)))
        logits = self.fc2(h)
        return logits, h, c


    def init_hidden(self):
        # return (torch.zeros(1, self.args.rnn_hidden_dim),
        #         torch.zeros(1, self.args.rnn_hidden_dim))
    
        return (self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_(),
                self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_())
