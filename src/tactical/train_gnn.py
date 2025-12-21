"""Training script for Graph Neural Network tactical models.

This script provides templates and utilities for training GNN models
on tactical analysis tasks.

TRAINING REQUIREMENTS:
1. Labeled tracking data with tactical states
2. PyTorch and PyTorch Geometric installed
3. GPU recommended for faster training
4. Data format: List of (graph, label) pairs

DATASETS:
- Recommended: StatsBomb, Metrica Sports, or custom labeled data
- Minimum: ~1000 labeled frames for basic training
- Recommended: 10k+ frames for production models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json

from .graph_builder import TrackingGraphBuilder
from .gnn_model import TacticalGNN, StateClassificationHead, create_tactical_gnn
from .team_state import TacticalState


class TacticalGraphDataset(Dataset):
    """Dataset for tactical graph classification."""

    def __init__(self, data_path: str, graph_builder: TrackingGraphBuilder):
        """
        Initialize dataset.

        Args:
            data_path: Path to labeled tracking data (JSON)
            graph_builder: Graph builder instance
        """
        self.graph_builder = graph_builder

        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Build graph from frame data
        frame_data = sample['frame_data']
        graph = self.graph_builder.build_graph(frame_data)

        # Add label
        label = sample['label']  # TacticalState enum value
        graph.y = torch.tensor([label], dtype=torch.long)

        return graph


def create_synthetic_training_data(num_samples: int = 1000, output_path: str = 'synthetic_tactical_data.json'):
    """
    Create synthetic training data for demonstration.

    In practice, use real labeled data from video analysis.

    Args:
        num_samples: Number of samples to generate
        output_path: Output file path
    """
    print(f"Generating {num_samples} synthetic training samples...")

    data = []

    for i in range(num_samples):
        # Random scenario
        scenario = np.random.choice(['attacking', 'defending', 'transition_attack', 'transition_defense', 'set_piece'])

        # Generate positions based on scenario
        if scenario == 'attacking':
            # Team 0 attacking (positions shifted toward right goal)
            team0_x = np.random.uniform(-10, 40, 6)
            team1_x = np.random.uniform(-5, 30, 6)
            ball_x = np.random.uniform(10, 45)
            label = TacticalState.ATTACKING.value

        elif scenario == 'defending':
            # Team 0 defending
            team0_x = np.random.uniform(-45, -10, 6)
            team1_x = np.random.uniform(-30, 5, 6)
            ball_x = np.random.uniform(-45, -10)
            label = TacticalState.DEFENDING.value

        elif scenario == 'transition_attack':
            # Counter-attack - high velocities
            team0_x = np.random.uniform(-20, 30, 6)
            team1_x = np.random.uniform(-30, 20, 6)
            ball_x = np.random.uniform(0, 30)
            label = TacticalState.TRANSITION_ATTACK.value

        elif scenario == 'transition_defense':
            # Losing possession
            team0_x = np.random.uniform(-30, 20, 6)
            team1_x = np.random.uniform(-20, 30, 6)
            ball_x = np.random.uniform(-20, 10)
            label = TacticalState.TRANSITION_DEFENSE.value

        else:  # set_piece
            # Set piece - clustered positions
            team0_x = np.random.normal(10, 5, 6)
            team1_x = np.random.normal(5, 5, 6)
            ball_x = np.random.uniform(-10, 10)
            label = TacticalState.SET_PIECE.value

        # Generate frame data
        players = []

        # Team 0
        for j in range(6):
            players.append({
                'track_id': j,
                'x': float(team0_x[j]),
                'y': float(np.random.uniform(-30, 30)),
                'vx': float(np.random.normal(0, 1) if scenario.startswith('transition') else np.random.normal(0, 0.3)),
                'vy': float(np.random.normal(0, 0.5)),
                'team': 0
            })

        # Team 1
        for j in range(6):
            players.append({
                'track_id': j + 6,
                'x': float(team1_x[j]),
                'y': float(np.random.uniform(-30, 30)),
                'vx': float(np.random.normal(0, 1) if scenario.startswith('transition') else np.random.normal(0, 0.3)),
                'vy': float(np.random.normal(0, 0.5)),
                'team': 1
            })

        frame_data = {
            'players': players,
            'ball': {'x': float(ball_x), 'y': float(np.random.uniform(-15, 15))}
        }

        data.append({
            'frame_data': frame_data,
            'label': label,
            'scenario': scenario
        })

    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved synthetic data to {output_path}")

    # Print distribution
    labels = [d['label'] for d in data]
    for state in TacticalState:
        count = labels.count(state.value)
        print(f"  {state.name}: {count} samples ({count/len(labels)*100:.1f}%)")


def train_tactical_gnn(
    train_dataset: TacticalGraphDataset,
    val_dataset: TacticalGraphDataset,
    config: Optional[Dict] = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: str = 'checkpoints/tactical_gnn.pt'
):
    """
    Train tactical GNN model.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Model configuration
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Training device
        save_path: Model checkpoint save path
    """
    print("="*60)
    print("Training Tactical GNN")
    print("="*60)
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print()

    # Create data loaders
    train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    gnn_model = create_tactical_gnn(config).to(device)
    classifier = StateClassificationHead(
        input_dim=gnn_model.output_dim,
        num_states=len(TacticalState)
    ).to(device)

    print(f"GNN parameters: {sum(p.numel() for p in gnn_model.parameters()):,}")
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    print()

    # Optimizer and loss
    optimizer = optim.Adam(
        list(gnn_model.parameters()) + list(classifier.parameters()),
        lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training
        gnn_model.train()
        classifier.train()
        train_loss = 0.0
        train_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            batch = batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            _, graph_embedding = gnn_model(batch)
            logits = classifier(graph_embedding)

            # Loss
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation
        val_loss, val_acc = evaluate(gnn_model, classifier, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(gnn_model, classifier, optimizer, epoch, save_path, config)

        # History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val Acc: {val_acc:.4f} - "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")

    return history


def evaluate(model, classifier, data_loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    classifier.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            _, graph_embedding = model(batch)
            logits = classifier(graph_embedding)

            loss = criterion(logits, batch.y.squeeze())
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch.y.squeeze()).sum().item()
            total += batch.y.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def save_checkpoint(model, classifier, optimizer, epoch, path, config):
    """Save model checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }

    torch.save(checkpoint, path)


def load_checkpoint(path, device='cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get('config', {})
    model = create_tactical_gnn(config).to(device)
    classifier = StateClassificationHead(
        input_dim=model.output_dim,
        num_states=len(TacticalState)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])

    model.eval()
    classifier.eval()

    return model, classifier


if __name__ == '__main__':
    """Example training pipeline."""

    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRAIN_DATA_PATH = 'data/train_tactical.json'
    VAL_DATA_PATH = 'data/val_tactical.json'

    # Step 1: Generate synthetic data (for demonstration)
    print("Step 1: Generating synthetic training data...")
    create_synthetic_training_data(num_samples=800, output_path=TRAIN_DATA_PATH)
    create_synthetic_training_data(num_samples=200, output_path=VAL_DATA_PATH)

    # Step 2: Create datasets
    print("\nStep 2: Creating datasets...")
    graph_builder = TrackingGraphBuilder()
    train_dataset = TacticalGraphDataset(TRAIN_DATA_PATH, graph_builder)
    val_dataset = TacticalGraphDataset(VAL_DATA_PATH, graph_builder)

    # Step 3: Train model
    print("\nStep 3: Training model...")
    config = {
        'input_dim': 12,
        'hidden_dim': 128,
        'num_layers': 4,
        'output_dim': 64,
        'gnn_type': 'gat',
        'num_heads': 4
    }

    history = train_tactical_gnn(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        num_epochs=30,
        batch_size=16,
        lr=0.001,
        device=DEVICE,
        save_path='checkpoints/tactical_gnn_best.pt'
    )

    print("\nTraining complete!")
