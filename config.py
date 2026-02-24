import os
import torch


class Config:
    """Configuration class for LPR training."""
    
    # Data paths
    DATA_ROOT = "/Users/nguyenthientai/Documents/baseline_icpr_2026-main/train 2"
    VAL_SPLIT_FILE = "val_tracks.json"
    
    # Image settings
    IMG_HEIGHT = 32
    IMG_WIDTH = 128
    
    # Character set
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
    
    # Training hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50
    SEED = 42
    # Note: Set NUM_WORKERS=0 for CPU to avoid multiprocessing issues
    NUM_WORKERS = 0  # 0 for CPU, can increase for GPU if needed
    PIN_MEMORY = False  # False for CPU, True for GPU
    
    # Device - Auto-detect CUDA or CPU
    # Note: CTCLoss doesn't support MPS (Apple Silicon), so we use CPU instead
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        USE_CUDA = True
        USE_CPU = False
    else:
        DEVICE = torch.device('cpu')
        USE_CUDA = False
        USE_CPU = True
    
    # Character mappings (computed from CHARS)
    CHAR2IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
    IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
    NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank
