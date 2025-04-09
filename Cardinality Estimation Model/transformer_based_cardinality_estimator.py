import torch
from torch import nn
import numpy as np
import utils


class Path_Cardinality_Estimator(nn.Module):
    def __init__(self, num_layers,hidden_size, device, alphabet_size, input_size):
        super(Path_Cardinality_Estimator, self).__init__()
        self.num_layers = num_layers  # Store num_layers passed to the model
        self.hidden_size = hidden_size
        self.device = device
        self.alphabet_size = alphabet_size
        self.input_size = input_size
        layer_sizes = [128, 64, 32, 16, 8]

        # Positional Encoding
        self.positional_encoding = self.generate_positional_encoding(6, hidden_size).to(device)
        self.input_projection = nn.Linear(input_size, hidden_size)
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=16,
            dim_feedforward=256,
            dropout=0.1,
            activation='relu'
        )
        # Now passing the num_layers as a parameter
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
            nn.ReLU(),
            nn.Linear(layer_sizes[3], layer_sizes[4]),
            nn.ReLU(),
            nn.Linear(layer_sizes[4], 1),
        )

    def generate_positional_encoding(self, seq_len, hidden_size):
        """Generates sinusoidal positional encodings."""
        pos = torch.arange(seq_len).unsqueeze(1)
        i = torch.arange(hidden_size).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / hidden_size)
        encodings = pos * angle_rates
        encodings[:, 0::2] = torch.sin(encodings[:, 0::2])
        encodings[:, 1::2] = torch.cos(encodings[:, 1::2])
        return encodings.unsqueeze(0)

    def forward_selectivity(self, x):
        # Project input to the correct hidden size
        x = self.input_projection(x)

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)

        # Pass through Transformer
        x = self.transformer(x)

        # Pass through Fully Connected layers
        x = torch.sigmoid(self.fc(x))
        return torch.squeeze(x)

    def forward(self, x):
        output = self.forward_selectivity(x)
        return output


def train_model(train_data, model, device, learning_rate, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        loss_list = []
        for i, (name, mask, target) in enumerate(train_data):
            name = name.to(device)
            output = model(name)
            target = target.to(device)
            mask = mask.to(device)

            # Compute Loss
            loss = utils.binary_crossentropy(output, target, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())

        if epoch % 10 == 0:
            print("Epoch: {}/{} - Mean Running Loss: {:.4f}".format(epoch + 1, num_epochs, np.mean(loss_list)))

    return model
