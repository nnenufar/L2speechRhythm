import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import AttentionPooling, Conv1dStack, LSTMEncoder, MLP


class RhythmEncoder(nn.Module):
    """CNN + bidirectional LSTM + attention pooling for envelope sequences."""

    def __init__(self,
                 feat_dim=1,
                 kernel_size=3,
                 cnn_out_channels=64,
                 num_cnn_layers=3,
                 cnn_stride=1,
                 lstm_hidden_size=64,
                 lstm_num_layers=1,
                 lstm_bidirectional=True,
                 attn_hidden_dim=None,
                 attn_pool_mode='sigmoid',
                 attn_tau=0.3,
                 dropout=0.1):
        super().__init__()

        self.attn_tau = attn_tau

        self.cnn_encoder = Conv1dStack(
            in_channels=feat_dim,
            hidden_channels=cnn_out_channels * 2,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            num_layers=num_cnn_layers,
            activation='relu',
            dropout=dropout,
            norm='layer',
            stride=cnn_stride,
        )

        self.lstm_encoder = LSTMEncoder(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
            bidirectional=lstm_bidirectional,
            output_mode='all',
        )

        self.lstm_attn_pool = AttentionPooling(
            embed_dim=self.lstm_encoder.output_size,
            hidden_dim=attn_hidden_dim,
            dropout=dropout,
            mode=attn_pool_mode,
        )

        self.embed_dim = self.lstm_encoder.output_size
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, return_attention=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.permute(0, 2, 1)

        cnn_out = self.cnn_encoder(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out = self.lstm_encoder(cnn_out)
        embeddings, attn_weights = self.lstm_attn_pool(lstm_out, tau=self.attn_tau)
        embeddings = self.layer_norm(embeddings)

        if return_attention:
            return embeddings, attn_weights
        return embeddings


class RhythmRegressor(nn.Module):
    """RhythmEncoder + regression head for envelope/envelope-derivative models."""

    def __init__(self, pretrained_checkpoint=None, freeze_encoder=True,
                 hidden_size=32, dropout=0.1, main_item='envelope', **kwargs):
        super().__init__()

        self.main_item = main_item

        encoder_params = inspect.signature(RhythmEncoder.__init__).parameters.keys()
        encoder_kwargs = {k: v for k, v in kwargs.items() if k in encoder_params}
        self.encoder = RhythmEncoder(**encoder_kwargs)

        if pretrained_checkpoint is not None:
            checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')
            encoder_state = {
                k.removeprefix('encoder.'): v
                for k, v in checkpoint['model_state_dict'].items()
                if k.startswith('encoder.')
            }
            self.encoder.load_state_dict(encoder_state, strict=False)

        if freeze_encoder is True or freeze_encoder == 'encoder_all':
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif freeze_encoder == 'cnn_only':
            for p in self.encoder.cnn_encoder.parameters():
                p.requires_grad = False

        self.head = MLP(
            in_features=self.encoder.embed_dim,
            hidden_features=hidden_size,
            out_features=1,
            num_layers=1,
            activation='relu',
            dropout=dropout,
            norm='layer',
        )

    def forward(self, batch, return_attention=False):
        x = batch[self.main_item]
        if return_attention:
            emb, attn = self.encoder(x, return_attention=True)
        else:
            emb = self.encoder(x)

        out = self.head(emb).squeeze(-1)
        if return_attention:
            return out, attn
        return out


class DurationRegressor(nn.Module):
    """Vocalic/intervocalic duration sequences -> LSTM -> attention pooling -> regression."""

    def __init__(self, num_tokens, phone_embed_dim=64, lstm_hidden_size=64,
                 lstm_num_layers=1, dropout=0.1, hidden_size=32, max_phones=5,
                 use_phone_durs_z=False, **kwargs):
        super().__init__()

        self.max_phones = max_phones
        self.use_phone_durs_z = use_phone_durs_z
        input_dim = (phone_embed_dim + 1) * max_phones + 1
        pool_embed = lstm_hidden_size * 2

        self.phone_embedding = nn.Embedding(num_tokens, phone_embed_dim, padding_idx=0)

        self.lstm_v = LSTMEncoder(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
            bidirectional=True,
            output_mode='all',
        )
        self.lstm_c = LSTMEncoder(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
            bidirectional=True,
            output_mode='all',
        )
        self.ln_v = nn.LayerNorm(input_dim)
        self.ln_c = nn.LayerNorm(input_dim)

        self.attn_pool_v = AttentionPooling(
            embed_dim=pool_embed, hidden_dim=None, dropout=dropout, mode='sigmoid'
        )
        self.attn_pool_c = AttentionPooling(
            embed_dim=pool_embed, hidden_dim=None, dropout=dropout, mode='sigmoid'
        )

        self.head = MLP(
            in_features=pool_embed * 2,
            hidden_features=hidden_size,
            out_features=1,
            num_layers=1,
            activation='relu',
            dropout=dropout,
            norm='layer',
        )

    def _embed_intervals(self, phones, phone_durs, dur):
        L = phones.size(-1)
        emb = self.phone_embedding(phones)

        if L < self.max_phones:
            emb = F.pad(emb, (0, 0, 0, self.max_phones - L))
            phone_durs = F.pad(phone_durs, (0, self.max_phones - L))
        elif L > self.max_phones:
            emb = emb[:, :, :self.max_phones, :]
            phone_durs = phone_durs[:, :, :self.max_phones]

        phone_durs = phone_durs.clamp(-5, 5)
        subseg = torch.cat([emb, phone_durs.unsqueeze(-1)], dim=-1)
        flat = subseg.reshape(*subseg.shape[:-2], -1)
        return torch.cat([flat, dur.unsqueeze(-1)], dim=-1)

    def forward(self, batch, return_attention=False):
        v_dur_key = 'v_phone_durs_z' if self.use_phone_durs_z else 'v_phone_durs'
        c_dur_key = 'c_phone_durs_z' if self.use_phone_durs_z else 'c_phone_durs'
        v_feat = self._embed_intervals(batch['v_phones'], batch[v_dur_key], batch['v_dur'])
        c_feat = self._embed_intervals(batch['c_phones'], batch[c_dur_key], batch['c_dur'])

        v_lengths = (batch['v_dur'] != 0).sum(dim=1).clamp(min=1)
        c_lengths = (batch['c_dur'] != 0).sum(dim=1).clamp(min=1)

        v_enc = self.lstm_v(self.ln_v(v_feat), v_lengths)
        c_enc = self.lstm_c(self.ln_c(c_feat), c_lengths)

        h_v, attn_v = self.attn_pool_v(v_enc)
        h_c, attn_c = self.attn_pool_c(c_enc)
        out = self.head(torch.cat([h_v, h_c], dim=-1)).squeeze(-1)

        if return_attention:
            return out, {'v': attn_v, 'c': attn_c}
        return out
