import torch.nn as nn
import torch.nn.functional as F


class SSDAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(SSDAgent, self).__init__()
        # input_shape, tensorflow中(height, width, channels)
        # torch中(channels, height, width)
        # 现在的输入是 (channels, height, width)
        self.args = args

        self.conv = nn.Sequential(
        nn.Conv2d(in_channels=input_shape[0], out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.Tanh(),
        nn.Flatten()
        )
        self.fc1 = nn.Linear(args.out_channels * (input_shape[1] - 2) * (input_shape[2] - 2), args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        inputs = inputs / 255.0
        # 输入绝大多数都是0呀！
        if inputs.device != hidden_state.device:
            inputs = inputs.to(hidden_state.device)
        x = self.conv(inputs)
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
