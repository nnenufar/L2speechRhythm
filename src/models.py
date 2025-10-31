import torch
import torch.nn.functional as F
import torch.nn as nn

class CNN_RNN_Classifier(nn.Module):
    """
    A hybrid model combining CNN for feature extraction and RNN for sequence modeling.
    Includes interval concatenation for temporal context awareness.
    """
    def __init__(self,
                 feat_dim,
                 kernel_size,
                 hidden_size,
                 num_classes, 
                 cnn_out_channels=None,
                 num_lstm_layers=2,
                 dropout=0.1):
        super(CNN_RNN_Classifier, self).__init__()

        self.cnn_out_channels = cnn_out_channels if cnn_out_channels else feat_dim * 2
        self.dropout = nn.Dropout(dropout)

        self.conv1d_a = nn.Conv1d(
            in_channels=feat_dim, 
            out_channels=self.cnn_out_channels,
            kernel_size=kernel_size,
            padding='valid' # No padding, already performed by the collate function 
        )
        self.batch_norm_a = nn.BatchNorm1d(num_features=self.cnn_out_channels)

        self.conv1d_b = nn.Conv1d(
            in_channels=self.cnn_out_channels, 
            out_channels=self.cnn_out_channels,
            kernel_size=kernel_size,
            padding='valid' 
        )
        self.batch_norm_b = nn.BatchNorm1d(num_features=self.cnn_out_channels)

        # LSTM layer to process the sequence of features from the CNN
        # Input size is CNN output channels + 1 (for the interval feature)
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels + 1,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, batch):
        """
        Args:
            x: Input MFCC features of shape (batch_size, seq_len, feat_dim)
            interval: Float tensor of shape (batch_size, seq_len)
        
        Returns:
            out: Classification logits of shape (batch_size, num_classes)
        """
        # CNN processing
        x = batch['feats']
        intervals = batch['intervals']
        x = x.permute(0, 2, 1)    # Conv1d receives (batch_size, feat_dim, seq_len)
        #print(f'Original Feats shape: {x.shape}')
        cnn_out = F.relu(self.conv1d_a(x))  # (batch_size, cnn_out_channels, seq_len)
        cnn_out = self.dropout(self.batch_norm_a(cnn_out))

        cnn_out = F.relu(self.conv1d_b(cnn_out))
        cnn_out = self.dropout(self.batch_norm_b(cnn_out))  
        
        # Reshape CNN feats for LSTM: (batch_size, seq_len, cnn_out_channels)
        rnn_input = cnn_out.permute(0, 2, 1)
        #print(f'RNN input: {rnn_input[0]}')
        #print(f'shape: {rnn_input.shape}')

        # Reshape intervals for concat
        intervals = intervals.unsqueeze(2)
        #print(f'Intervals shape: {intervals.shape}')
        
        # Concatenate interval with CNN features along the feature dimension
        rnn_input = torch.cat([rnn_input, intervals], dim=2)

        # LSTM processing
        lstm_out, (h_n, _) = self.lstm(rnn_input)
        # h_n shape: (num_layers, batch_size, hidden_size)

        last_hidden_state = h_n[-1]  # (batch_size, hidden_size)
        last_hidden_state = self.dropout(last_hidden_state)

        # Classification
        out = self.fc(last_hidden_state)  # (batch_size, num_classes)

        return out
    
class LSTM_with_MultiHeadAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 pair_emb_dim,
                 num_classes,
                 dropout,
                 bidirectional,
                 input_size,
                 num_layers,
                 num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.embedding_dim = hidden_size * self.num_directions

        # --- LSTM backbone ---
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # --- Multi-head self-attention ---
        # Note: embedding_dim must be divisible by num_heads
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # --- Feed-forward classification head ---
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, pair_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_emb_dim, num_classes)
        )

        # Optional: normalization for stability
        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.norm2 = nn.LayerNorm(self.embedding_dim)

    def forward(self, batch):
        freqs = batch['spectrum_freq_bins']
        env_spec = batch['envelope_spectrum']
        x = torch.stack([freqs, env_spec], dim=-1)  # (B, T, 2)

        # --- LSTM encoding ---
        lstm_out, _ = self.lstm(x)  # (B, T, H*D)
        lstm_out = self.norm1(lstm_out)

        # --- Multi-head self-attention ---
        attn_out, attn_weights = self.self_attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm2(attn_out + lstm_out)  # residual connection

        # --- Pooling ---
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # --- Classification ---
        out = self.fc(pooled)

        return out # attn_weights can be visualized later
    
class rhythm_spectrum_encoder(nn.Module):
    def __init__(self,
                posEmb_dim,
                cnn_out_channels,
                cnn_kernel_size,
                lstm_num_layers,
                lstm_hidden_size,
                dropout,
                max_audio_dur = 30,
                sr=16000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        max_num_samples = sr * max_audio_dur
        max_num_freq_bins = int(10 * max_num_samples / sr) + 1
        print(f"Max num freq bins: {max_num_freq_bins}")

        # Encode freq bins as positional embeddings
        self.posEmb = nn.Embedding(max_num_freq_bins, posEmb_dim)

        self.conv1d_a = nn.Conv1d(
            in_channels=1, 
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
            padding='valid' # No padding, already performed by the collate function 
        )
        self.batch_norm_a = nn.BatchNorm1d(num_features=cnn_out_channels)

        self.conv1d_b = nn.Conv1d(
            in_channels=cnn_out_channels, 
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
            padding='valid' 
        )
        self.batch_norm_b = nn.BatchNorm1d(num_features=cnn_out_channels)

        # LSTM layer to process the sequence of features from the CNN
        # Input size is CNN output channels + 1 (for the interval feature)
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels + 1,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )

    def forward(self, batch):
        # CNN processing
        x = batch['envelope_spectrum'].unsqueeze(1)  # (batch_size, 1, num_freq_bins)
        # Conv1d receives (batch_size, feat_dim, seq_len)
        print(f'Original envSpec shape: {x.shape}')
        cnn_out = F.relu(self.conv1d_a(x))  # (batch_size, cnn_out_channels, seq_len)
        cnn_out = self.dropout(self.batch_norm_a(cnn_out))
        cnn_out = F.relu(self.conv1d_b(cnn_out))
        cnn_out = self.dropout(self.batch_norm_b(cnn_out))

        freq_bins = batch['spectrum_freq_bins']
        pos_embs = self.posEmb(freq_bins)
        
        # Reshape CNN feats for LSTM: (batch_size, seq_len, cnn_out_channels)
        rnn_input = cnn_out.permute(0, 2, 1)
        #print(f'RNN input: {rnn_input[0]}')
        #print(f'shape: {rnn_input.shape}')

        # Reshape intervals for concat
        pos_embs = pos_embs.unsqueeze(2)
        #print(f'Intervals shape: {intervals.shape}')
        
        # Concatenate interval with CNN features along the feature dimension
        rnn_input = torch.cat([rnn_input, pos_embs], dim=2)

        # LSTM processing
        lstm_out, (h_n, _) = self.lstm(rnn_input)
        # h_n shape: (num_layers, batch_size, hidden_size)

        last_hidden_state = h_n[-1]  # (batch_size, hidden_size)
        last_hidden_state = self.dropout(last_hidden_state)
