"""Utility functions for model operations"""

import torch
from typing import Dict, Any
from pathlib import Path

def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")
    
    return device

def count_parameters(model) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_training_history(history: Dict[str, list], save_path: str):
    """Save training history to JSON"""
    import json
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"✓ Training history saved to {save_path}")

def load_training_history(load_path: str) -> Dict[str, list]:
    """Load training history from JSON"""
    import json
    with open(load_path, 'r') as f:
        history = json.load(f)
    return history
