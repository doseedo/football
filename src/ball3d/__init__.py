"""3D Ball Trajectory Estimation Module.

This module provides complete 3D ball tracking capabilities including:
- Physics-based ball motion modeling
- Synthetic training data generation
- LSTM-based 3D trajectory estimation
- Combined 2D detection + 3D estimation tracker
"""

from .physics_model import (
    BallPhysicsModel,
    Ball3DPosition,
    PhysicsConstraints
)

from .synthetic_generator import (
    SyntheticDataGenerator,
    SyntheticTrajectory,
    CameraParameters
)

from .trajectory_lstm import (
    TrajectoryLSTM,
    CanonicalRepresentation,
    TrajectoryLoss,
    create_context_features,
    train_step,
    evaluate
)

from .ball3d_tracker import (
    Ball3DTracker,
    Ball3DState
)

__all__ = [
    # Physics model
    'BallPhysicsModel',
    'Ball3DPosition',
    'PhysicsConstraints',

    # Synthetic data
    'SyntheticDataGenerator',
    'SyntheticTrajectory',
    'CameraParameters',

    # Neural network
    'TrajectoryLSTM',
    'CanonicalRepresentation',
    'TrajectoryLoss',
    'create_context_features',
    'train_step',
    'evaluate',

    # Tracker
    'Ball3DTracker',
    'Ball3DState'
]

__version__ = '0.1.0'
