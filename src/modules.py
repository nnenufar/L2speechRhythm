import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', dropout=0.0, norm=None):
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]

        if norm == 'layer':
            layers.append(nn.LayerNorm(out_features))
        elif norm == 'batch':
            layers.append(nn.BatchNorm1d(out_features))

        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2,
                 activation='relu', dropout=0.0, norm=None):
        super().__init__()
        layers = [LinearBlock(in_features, hidden_features, activation, dropout, norm)]
        for _ in range(num_layers - 2):
            layers.append(LinearBlock(hidden_features, hidden_features, activation, dropout, norm))
        layers.append(nn.Linear(hidden_features, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same',
                 dilation=1, activation='relu', dropout=0.0, norm=None):
        super().__init__()

        effective_kernel = (kernel_size - 1) * dilation + 1
        if stride > 1:
            padding_val = effective_kernel // 2
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding_val, dilation=dilation,
            )
        else:
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
            )

        self.norm = None
        self.use_layer_norm = False
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'layer':
            self.use_layer_norm = True
            self.layer_norm = nn.LayerNorm(out_channels)

        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        elif self.use_layer_norm:
            x = x.permute(0, 2, 1)
            x = self.layer_norm(x)
            x = x.permute(0, 2, 1)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Conv1dStack(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size,
                 num_layers=2, activation='relu', dropout=0.0, norm='batch',
                 stride=1, dilation=1):
        super().__init__()
        strides = self._broadcast(stride, num_layers)
        dilations = [dilation * (2 ** i) for i in range(num_layers)]

        layers = [
            Conv1dBlock(
                in_channels, hidden_channels, kernel_size,
                stride=strides[0], dilation=dilations[0],
                activation=activation, dropout=dropout, norm=norm,
            )
        ]
        for i in range(1, num_layers - 1):
            layers.append(Conv1dBlock(
                hidden_channels, hidden_channels, kernel_size,
                stride=strides[i], dilation=dilations[i],
                activation=activation, dropout=dropout, norm=norm,
            ))
        if num_layers > 1:
            layers.append(Conv1dBlock(
                hidden_channels, out_channels, kernel_size,
                stride=strides[-1], dilation=dilations[-1],
                activation=activation, dropout=dropout, norm=norm,
            ))

        self.stack = nn.Sequential(*layers)

    @staticmethod
    def _broadcast(stride, num_layers):
        if isinstance(stride, int):
            return (stride,) * num_layers
        s = tuple(stride)
        if len(s) != num_layers:
            raise ValueError(f"stride tuple length {len(s)} != num_layers {num_layers}")
        return s

    def forward(self, x):
        return self.stack(x)


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, dropout=0.0, mode='sigmoid', use_norm=True):
        super().__init__()
        assert mode in ('sigmoid', 'softmax')
        self.mode = mode
        self.use_norm = use_norm

        if hidden_dim is None:
            self.score = nn.Linear(embed_dim, 1)
        else:
            self.score = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        self.norm = nn.LayerNorm(embed_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, mask=None, tau=0.3):
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))

        if self.mode == 'softmax':
            attn = F.softmax(logits / tau, dim=-1)
        else:
            g = torch.sigmoid(logits / tau)
            attn = g / (g.sum(dim=1, keepdim=True) + 1e-8)

        if self.dropout is not None:
            attn = self.dropout(attn)

        pooled = torch.einsum('btc,bt->bc', x, attn)
        pooled = self.norm(pooled)
        return pooled, attn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0,
                 bidirectional=True, output_mode='all'):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_size = hidden_size * self.num_directions
        self.output_mode = output_mode

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(self.output_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, lengths=None):
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            lstm_out, (h_n, _) = self.lstm(x_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, _) = self.lstm(x)

        if self.output_mode == 'last':
            if self.bidirectional:
                output = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                output = h_n[-1]
        elif self.output_mode == 'pool':
            if lengths is not None:
                mask = torch.arange(lstm_out.size(1), device=lstm_out.device).unsqueeze(0) < lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).float()
                output = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                output = lstm_out.mean(dim=1)
        else:
            output = lstm_out

        output = self.layer_norm(output)
        if self.dropout is not None:
            output = self.dropout(output)
        return output
