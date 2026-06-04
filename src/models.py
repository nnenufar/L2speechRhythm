import torch
import inspect
from transformers import WavLMModel
import torch.nn.functional as F
import torch.nn as nn
from src.modules import *
from src.train_utils import quantize_values


class RhythmEncoder(nn.Module):
    """
    Encoder for amplitude envelope sequences.
    CNN + LSTM → attention pooling → embedding
    """
    def __init__(self,
                 num_phonemes=0,
                 use_text=False,
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
                 attn_tau=2.0,
                 dropout=0.1):
        super().__init__()

        self.attn_tau = attn_tau
        self.use_text = use_text
        audio_dim = lstm_hidden_size * (2 if lstm_bidirectional else 1)

        self.cnn_encoder = Conv1dStack(
            in_channels=feat_dim,
            hidden_channels=cnn_out_channels,
            out_channels=cnn_out_channels,
            kernel_size=kernel_size,
            num_layers=num_cnn_layers,
            activation='relu',
            dropout=dropout,
            norm='layer',
            stride=cnn_stride
        )

        self.lstm_encoder = LSTMEncoder(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
            bidirectional=lstm_bidirectional,
            output_mode='all'
        )

        if use_text:
            self.text_encoder = TextEncoder(
                vocab_size=num_phonemes,
                embed_dim=64,
                conv_channels=128,
                kernel_size=5,
                lstm_hidden=lstm_hidden_size,
                lstm_layers=lstm_num_layers,
                dropout=dropout
            )

            text_dim = self.text_encoder.output_dim

            self.cross_attn = AdditiveCrossAttention(
                query_dim=audio_dim,
                key_dim=text_dim,
                attn_dim=64,
                dropout=dropout
            )

            self.cross_attn_proj = nn.Linear(audio_dim + text_dim, audio_dim)

        self.lstm_attn_pool = AttentionPooling(
            embed_dim=audio_dim,
            hidden_dim=attn_hidden_dim,
            dropout=dropout,
            mode=attn_pool_mode
        )
        self.embed_dim = audio_dim

        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, phoneme_ids=None, phoneme_lengths=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        cnn_out = self.cnn_encoder(x)
        cnn_out = cnn_out.permute(0, 2, 1)

        audio_features = self.lstm_encoder(cnn_out)

        if self.use_text:
            text_features = self.text_encoder(phoneme_ids, phoneme_lengths)
            attended, attn_weights = self.cross_attn(audio_features, text_features, phoneme_lengths)
            enriched = torch.cat([audio_features, attended], dim=-1)
            audio_features = self.cross_attn_proj(enriched)
        else:
            attn_weights = None

        embeddings, _ = self.lstm_attn_pool(audio_features, tau=self.attn_tau)

        embeddings = self.layer_norm(embeddings)
        return embeddings, attn_weights


class RhythmContrastiveModel(nn.Module):
    """
    Rhythm encoder + projection head for contrastive pretraining.
    Forward pass: encoder → MLP projection → L2-normalize.
    The encoder output is the representation used for downstream tasks.
    """
    def __init__(self, proj_hidden_dim=128, proj_out_dim=64, proj_num_layers=2,
                 proj_dropout=0.0, **encoder_kwargs):
        super().__init__()
        self.encoder = RhythmEncoder(**encoder_kwargs)
        self.projection = MLP(
            in_features=self.encoder.embed_dim,
            hidden_features=proj_hidden_dim,
            out_features=proj_out_dim,
            num_layers=proj_num_layers,
            activation='relu',
            dropout=proj_dropout,
            norm=None
        )

    def forward(self, x, phoneme_ids=None, phoneme_lengths=None):
        embeddings, attn_weights = self.encoder(x, phoneme_ids, phoneme_lengths)
        projected = self.projection(embeddings)
        return F.normalize(projected, dim=-1), attn_weights


class RhythmRegressor(nn.Module):
    """
    Pretrained RhythmEncoder + regression head for fluency score prediction.
    Loads encoder weights from a contrastive checkpoint, adds an MLP head.
    """
    def __init__(self, pretrained_checkpoint=None, freeze_encoder=True,
                 hidden_size=32, dropout=0.1, **kwargs):
        super().__init__()

        encoder_params = inspect.signature(RhythmEncoder.__init__).parameters.keys()
        encoder_kwargs = {k: v for k, v in kwargs.items() if k in encoder_params}
        self.encoder = RhythmEncoder(**encoder_kwargs)

        if pretrained_checkpoint is not None:
            checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')
            encoder_state = {k.removeprefix('encoder.'): v
                             for k, v in checkpoint['model_state_dict'].items()
                             if k.startswith('encoder.')}
            self.encoder.load_state_dict(encoder_state, strict=False)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.use_text = self.encoder.use_text

        self.head = MLP(
            in_features=self.encoder.embed_dim,
            hidden_features=hidden_size,
            out_features=1,
            num_layers=2,
            activation='relu',
            dropout=dropout,
            norm='layer'
        )

    def forward(self, batch, return_attention=False):
        x = batch['envelope']

        if self.use_text:
            emb, attn_weights = self.encoder(
                x, batch['phoneme_ids'], batch['phoneme_lengths']
            )
        else:
            emb, attn_weights = self.encoder(x)

        out = self.head(emb).squeeze(-1)

        if return_attention and attn_weights is not None:
            return out, attn_weights
        return out


class DurationRegressor(nn.Module):
    """
    Vocalic + intervocalic interval sequences → LSTM → attention pooling → regression.
    Multi-phone intervals are mean-pooled from individual phone embeddings.
    """
    def __init__(self, num_tokens, phone_embed_dim=64, lstm_hidden_size=64,
                 lstm_num_layers=1, dropout=0.1, hidden_size=32, **kwargs):
        super().__init__()

        input_dim = phone_embed_dim + 1
        lstm_out = lstm_hidden_size * 2

        self.phone_embedding = nn.Embedding(num_tokens, phone_embed_dim, padding_idx=0)

        self.lstm_v = LSTMEncoder(
            input_size=input_dim, hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers, dropout=dropout,
            bidirectional=True, output_mode='all'
        )
        self.attn_pool_v = AttentionPooling(
            embed_dim=lstm_out, hidden_dim=None, dropout=dropout, mode='sigmoid'
        )

        self.lstm_c = LSTMEncoder(
            input_size=input_dim, hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers, dropout=dropout,
            bidirectional=True, output_mode='all'
        )
        self.attn_pool_c = AttentionPooling(
            embed_dim=lstm_out, hidden_dim=None, dropout=dropout, mode='sigmoid'
        )

        self.head = MLP(
            in_features=lstm_out * 2, hidden_features=hidden_size,
            out_features=1, num_layers=1, activation='relu',
            dropout=dropout, norm='layer'
        )

    def _embed_intervals(self, phones, plen, dur):
        emb = self.phone_embedding(phones)
        mask = torch.arange(phones.size(-1), device=phones.device).unsqueeze(0).unsqueeze(0)
        mask = (mask < plen.unsqueeze(-1)).float().unsqueeze(-1)
        pooled = (emb * mask).sum(dim=2) / plen.unsqueeze(-1).clamp(min=1)
        return torch.cat([pooled, dur.unsqueeze(-1)], dim=-1)

    def forward(self, batch, return_attention=False):
        v_feat = self._embed_intervals(batch['v_phones'], batch['v_plen'], batch['v_dur'])
        c_feat = self._embed_intervals(batch['c_phones'], batch['c_plen'], batch['c_dur'])

        v_lengths = (batch['v_dur'] != 0).sum(dim=1).clamp(min=1)
        c_lengths = (batch['c_dur'] != 0).sum(dim=1).clamp(min=1)

        v_lstm = self.lstm_v(v_feat, v_lengths)
        h_v, _ = self.attn_pool_v(v_lstm)

        c_lstm = self.lstm_c(c_feat, c_lengths)
        h_c, _ = self.attn_pool_c(c_lstm)

        out = self.head(torch.cat([h_v, h_c], dim=-1)).squeeze(-1)

        if return_attention:
            return out, None
        return out


class DurationContrastiveModel(nn.Module):
    """
    Duration-based contrastive pretraining: V/C duration sequences → LSTM → projection.
    No phone identities — input is scalar duration per interval.
    """
    def __init__(self, lstm_hidden_size=64, lstm_num_layers=1, dropout=0.1,
                 proj_hidden_dim=128, proj_out_dim=64, **kwargs):
        super().__init__()

        input_dim = 1
        lstm_out = lstm_hidden_size * 2

        self.lstm_v = LSTMEncoder(
            input_size=input_dim, hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers, dropout=dropout,
            bidirectional=True, output_mode='all'
        )
        self.attn_pool_v = AttentionPooling(
            embed_dim=lstm_out, hidden_dim=None, dropout=dropout, mode='sigmoid'
        )

        self.lstm_c = LSTMEncoder(
            input_size=input_dim, hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers, dropout=dropout,
            bidirectional=True, output_mode='all'
        )
        self.attn_pool_c = AttentionPooling(
            embed_dim=lstm_out, hidden_dim=None, dropout=dropout, mode='sigmoid'
        )

        embed_dim = lstm_out * 2
        self.projection = MLP(
            in_features=embed_dim, hidden_features=proj_hidden_dim,
            out_features=proj_out_dim, num_layers=2,
            activation='relu', dropout=dropout, norm=None
        )

    def forward(self, v_dur, c_dur):
        v_feat = v_dur.unsqueeze(-1)
        c_feat = c_dur.unsqueeze(-1)

        v_lengths = (v_dur != 0).sum(dim=1).clamp(min=1)
        c_lengths = (c_dur != 0).sum(dim=1).clamp(min=1)

        v_lstm = self.lstm_v(v_feat, v_lengths)
        h_v, _ = self.attn_pool_v(v_lstm)

        c_lstm = self.lstm_c(c_feat, c_lengths)
        h_c, _ = self.attn_pool_c(c_lstm)

        emb = torch.cat([h_v, h_c], dim=-1)
        projected = self.projection(emb)
        return F.normalize(projected, dim=-1)


class CNN_MLP(nn.Module):
    """
    CNN + LSTM + attention pooling with classification/regression head.
    """
    def __init__(self,
                 feat_dim,
                 kernel_size,
                 hidden_size,
                 num_classes,
                 cnn_out_channels=None,
                 num_cnn_layers=3,
                 attn_hidden_dim=None,
                 attn_pool_mode='sigmoid',
                 attn_tau=2.0,
                 dropout=0.1,
                 cnn_stride=1,
                 lstm_hidden_size=64,
                 lstm_num_layers=1,
                 lstm_bidirectional=True,
                 main_item=None,
                 f0_mean=None,
                 f0_std=None,
                 task='classification',
                 target_mean=None,
                 target_std=None,
                 num_utterances=0,
                 utterance_embed_dim=0):
        super().__init__()

        self.main_item = main_item
        self.cnn_out_channels = cnn_out_channels if cnn_out_channels else feat_dim * 2
        self.f0_mean = f0_mean
        self.f0_std = f0_std
        self.attn_tau = attn_tau
        self.task = task
        self.target_mean = target_mean
        self.target_std = target_std

        self.use_utterance_embedding = (utterance_embed_dim > 0 and num_utterances > 0)
        if self.use_utterance_embedding:
            self.utterance_embedding = nn.Embedding(num_utterances, utterance_embed_dim)
        self._utterance_embed_dim = utterance_embed_dim

        self.cnn_encoder = Conv1dStack(
            in_channels=feat_dim,
            hidden_channels=self.cnn_out_channels,
            out_channels=self.cnn_out_channels,
            kernel_size=kernel_size,
            num_layers=num_cnn_layers,
            activation='relu',
            dropout=dropout,
            norm='layer',
            stride=cnn_stride
        )

        self.lstm_encoder = LSTMEncoder(
            input_size=self.cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
            bidirectional=lstm_bidirectional,
            output_mode='all'
        )

        self.lstm_attn_pool = AttentionPooling(
            embed_dim=self.lstm_encoder.output_size,
            hidden_dim=attn_hidden_dim,
            dropout=dropout,
            mode=attn_pool_mode
        )
        self.embed_dim = self.lstm_encoder.output_size

        fc_in_features = self.embed_dim + utterance_embed_dim if self.use_utterance_embedding else self.embed_dim

        if task == 'regression':
            self.fc = MLP(
                in_features=fc_in_features,
                hidden_features=hidden_size,
                out_features=1,
                num_layers=1,
                activation='relu',
                dropout=dropout,
                norm='layer'
            )
        else:
            self.fc = MLP(
                in_features=fc_in_features,
                hidden_features=hidden_size,
                out_features=num_classes,
                num_layers=1,
                activation='relu',
                dropout=dropout,
                norm='layer'
            )

    def forward(self, batch, return_attention=False):
        x = batch[self.main_item]

        if self.main_item == 'f0':
            mask = batch['voiced_mask'].bool()
            f0_norm = torch.zeros_like(x)
            f0_norm[mask] = (x[mask] - self.f0_mean) / (self.f0_std + 1e-8)
            x = f0_norm

        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.permute(0, 2, 1)

        cnn_out = self.cnn_encoder(x)
        cnn_out = cnn_out.permute(0, 2, 1)

        lstm_out = self.lstm_encoder(cnn_out)
        embeddings, attn_weights = self.lstm_attn_pool(lstm_out, tau=self.attn_tau)

        if self.use_utterance_embedding:
            utterance_id = batch['utterance_id']
            utt_embed = self.utterance_embedding(utterance_id)
            embeddings = torch.cat([embeddings, utt_embed], dim=-1)

        logits = self.fc(embeddings)

        if self.task == 'regression':
            logits = logits.squeeze(-1)

        if return_attention and attn_weights is not None:
            return logits, attn_weights

        return logits
    

class WAV_LM(nn.Module):
    def __init__(self, model_name, num_classes, hidden_size=128, attn_hidden_dim=None, 
                 dropout=0.1, freeze_encoder=True, cache_dir=None, local_files_only=False,
                 attn_pool_mode='sigmoid', attn_tau=2.0, use_attention_pooling=True):
        super().__init__()
        self.attn_tau = attn_tau
        self.use_attn_pooling = use_attention_pooling
        self.model = WavLMModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        
        # Optionally freeze the encoder
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get hidden size from the model config
        self.encoder_dim = self.model.config.hidden_size  # 1024 for large
        
        # Attention pooling
        self.attn_pool = AttentionPooling(
            embed_dim=self.encoder_dim,
            hidden_dim=attn_hidden_dim,
            dropout=dropout,
            mode=attn_pool_mode
        )
        
        # Classification head
        self.fc = MLP(
            in_features=self.encoder_dim,
            hidden_features=hidden_size,
            out_features=num_classes,
            num_layers=1,
            activation='relu',
            dropout=dropout,
            norm='layer'
        )
    
    def forward(self, batch, return_attention=False):
        waveform = batch['waveform']  # (B, T_samples)
        
        # Get WavLM hidden states
        outputs = self.model(waveform)
        hidden_states = outputs.last_hidden_state  # (B, T_frames, D)

        if self.use_attn_pooling:
            # Attention pooling
            embeddings, attn_weights = self.attn_pool(hidden_states, tau=self.attn_tau)  # (B, D), (B, T_frames)
        else:
            embeddings = hidden_states.mean(dim=1)  # (B, D)
            attn_weights = None

        # Classification
        logits = self.fc(embeddings)
        
        if return_attention:
            return logits, attn_weights
        
        return logits