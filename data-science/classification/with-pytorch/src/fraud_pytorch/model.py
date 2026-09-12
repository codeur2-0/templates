from pathlib import Path

import torch
from torch import nn


class FraudMLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 12) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination)


class TorchTrainer:
    def __init__(self, model: FraudMLP, learning_rate: float, epochs: int) -> None:
        self.model = model
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss = nn.BCEWithLogitsLoss()

    def fit(self, loader) -> list[float]:
        history = []
        self.model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for features, target in loader:
                self.optimizer.zero_grad()
                loss = self.loss(self.model(features), target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.detach())
            history.append(epoch_loss / max(1, len(loader)))
        return history
