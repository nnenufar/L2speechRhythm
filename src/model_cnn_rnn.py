import torch.nn as nn

class CNN_RNN_Classifier(nn.Module):
    """
    A hybrid model combining CNN for feature extraction and RNN for sequence modeling.
    """
    def __init__(self, input_size, cnn_out_channels, kernel_size, hidden_size, num_classes):
        super(CNN_RNN_Classifier, self).__init__()

        # CNN layer to extract local features/motifs
        # input_size is the feature dimension at each time step (1 for raw data)
        self.conv1d = nn.Conv1d(in_channels=input_size,
                                out_channels=cnn_out_channels,
                                kernel_size=kernel_size)
        self.relu = nn.ReLU()
        #self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM layer to process the sequence of features from the CNN
        # Input size for the LSTM is the number of output channels from the CNN
        self.lstm = nn.LSTM(input_size=cnn_out_channels,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)

        # Final fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) # -> (batch_size, 1, seq_len)

        cnn_out = self.conv1d(x) # -> (batch_size, cnn_out_channels, new_seq_len)
        cnn_out = self.relu(cnn_out)
        #cnn_out = self.pool(cnn_out) # -> (batch_size, cnn_out_channels, even_newer_seq_len)

        rnn_input = cnn_out.permute(0, 2, 1) # -> (batch_size, even_newer_seq_len, cnn_out_channels)

        _, (h_n, _) = self.lstm(rnn_input) # h_n shape: (num_layers, batch_size, hidden_size)

        last_hidden_state = h_n[-1] # -> (batch_size, hidden_size)

        out = self.fc(last_hidden_state) # -> (batch_size, num_classes)

        return out