# MultiPriNTF: Privacy-Preserving Multimodal NFT Market Analysis

## Overview

MultiPriNTF is an advanced research framework for analyzing Non-Fungible Token (NFT) markets using privacy-preserving multimodal machine learning techniques. The system integrates visual and transaction data analysis with state-of-the-art privacy preservation mechanisms, enabling comprehensive market insights while protecting sensitive information.

## Key Features

- **Multimodal Fusion**: Integrates visual and transaction data
- **Privacy-Preserving Techniques**: Implements differential privacy
- **High Performance**: 99.98% parameter efficiency
- **Scalable Architecture**: Optimized for multi-GPU processing
- **Comprehensive Market Analysis**: Advanced NFT market prediction

## System Requirements

### Hardware
- Minimum: CUDA-capable GPU with 8GB VRAM
- Recommended: 3x NVIDIA RTX 3090 GPUs (24GB VRAM each)
- CPU: Intel/AMD x86_64 processor
- RAM: 128GB 

### Software
- Python 3.8+
- CUDA 11.7+
- PyTorch 2.1.0+
- Operating Systems: Linux (Recommended), Windows, macOS

## Installation

### 1. Clone the Repository


### 2. Create Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


### 3. Install Dependencies

pip install -r requirements.txt

## Data Preparation

### Downloading Dataset

The CryptoPunks dataset used in our research can be downloaded from:
- [https://www.kaggle.com/datasets/tunguz/cryptopunks]

Place the dataset in the `data/raw/` directory:
MultiPriNTF/
└── data/
    ├── raw/
    │   ├── images/
    │   └── transactions.jsonl
    └── processed/

### Preprocessing Data

python -m src.preprocessing \
    --image-dir data/raw/images \
    --transaction-file data/raw/transactions.jsonl \
    --output-dir data/processed

## Training

### Configuration

Modify `configs/config.yaml` to adjust training parameters:
- Set GPU configurations
- Adjust learning rates
- Configure privacy budget
- Customize model architecture

### Run Training

python -m src.main \
    --config configs/config.yaml \
    --experiment MultiPriNTF_run1 \
    --preprocess

### Training Options

- `--preprocess`: Preprocess data before training
- `--evaluate`: Run evaluation after training
- `--evaluate_only`: Perform only model evaluation

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:
- Market Efficiency Score
- Price Prediction Accuracy
- Feature Extraction Performance
- Privacy Budget Utilization

## Reproducibility

To ensure reproducibility:
- Fixed random seeds
- Deterministic CUDA operations
- Comprehensive logging
- Checkpoint management

## Security and Privacy

MultiPriNTF implements:
- Differential Privacy (ε < 0.5)
- Secure Feature Encryption
- Privacy Budget Tracking
- Gradient Perturbation

## Performance Metrics

- Processing Speed: 832 transactions/second
- Market Efficiency Score: 0.85
- Price Prediction Accuracy: 
  - Under 50 ETH: 92.7%
  - Over 50 ETH: 87.3%

## Limitations

- Performance may vary with different hardware
- Requires significant computational resources
- Limited to specific NFT collection characteristics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
