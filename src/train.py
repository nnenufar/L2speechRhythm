import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
from models import CNN_RNN_Classifier
from dataloaders import NpzDataset, collate_fn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config):
    dataset = NpzDataset(config['npz_path'], config['labels_path'])
    dataloader = DataLoader(dataset, batch_size = config['batch_size'], collate_fn = collate_fn, shuffle=True)
    num_classes = len(dataset.labels_str2int)

    model = config['model'](
        **config['model_params'],
        num_classes = num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training_params']['learning_rate']
    )

    print("\n--- Training started ---")
    for epoch in range(config['training_params']['epochs']):
        model.train()
        total_loss = 0
        for sequences_batch, labels_batch in dataloader:
            sequences_batch = sequences_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(sequences_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config['training_params']['epochs']}], Loss: {avg_loss:.4f}")
    #TODO: define checkpoint saving strategy

if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Train a sequence classification model.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the JSON configuration file.')
    args = parser.parse_args()

    # Load the configuration from the specified JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Run the training process
    train(config)