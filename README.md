# Graph - Multi-Relational Factor Graph Ranking System

## Overview

Graph is a sophisticated quantitative finance system that uses multi-relational graph neural networks for factor-based stock ranking and portfolio construction. The system combines multiple graph types (prior knowledge, factor similarity, dynamic behavior, and latent relationships) to learn comprehensive representations of financial factors and their interactions.

## Architecture

### Core Components

#### 1. Multi-Relational Factor Graph Ranker (`modeling/ranker.py`)
The main model that orchestrates the entire pipeline:
- **Feature Encoders**: Separate encoders for style factors, alpha factors, and temporal patterns
- **Graph Builders**: Construct different types of relational graphs
- **Relation Stack**: Multi-layer graph neural network for message passing
- **Semiring Composer**: Combines information from multiple relation types
- **Ranking Head**: Produces final stock rankings

#### 2. Graph Types (`graphs/`)

**Prior Graph** (`graphs/prior.py`)
- Encodes domain knowledge about factor relationships
- Industry-based and factor-type priors

**Factor Similarity Graph** (`graphs/similarity.py`)
- Learns similarity relationships between factors
- Top-k similarity connections

**Dynamic Behavior Graph** (`graphs/dynamic.py`)
- Captures time-varying correlations between assets
- Based on historical return patterns
- Uses correlation-based edge weights

**Latent Graph Learner** (`graphs/latent.py`)
- Learns hidden relationships from data
- Adaptive graph structure learning

#### 3. Data Pipeline (`data/`)

**Data Loading** (`data/io.py`)
- Loads factor data, labels, and liquidity information
- Handles wide-to-long format conversion

**Preprocessing** (`data/preprocess.py`)
- Style/alpha factor separation
- Cross-sectional standardization
- Industry code processing

**Dataset** (`data/dataset.py`)
- QuarterDataset for temporal slicing
- DataLoader for batch processing

**Split Strategy** (`data/split.py`)
- Time-based train/validation/test splits
- Season-aware data partitioning

#### 4. Training Framework (`training/`)

**Pipeline** (`training/pipeline.py`)
- End-to-end training for single seasons
- Ablation study support
- Model checkpointing

**Lightning Module** (`training/lightning_module.py`)
- PyTorch Lightning integration
- Automatic optimization and logging

#### 5. Evaluation System (`evaluation/`)

**Metrics** (`evaluation/metrics.py`)
- Information Coefficient (IC) and RankIC
- Portfolio return simulation
- NDCG@200 ranking quality
- Stability metrics

**Cross-Section Analysis** (`evaluation/cross_section.py`)
- Cross-sectional ranking metrics
- Statistical significance testing

**Portfolio Simulation** (`evaluation/portfolio.py`)
- Realistic portfolio construction
- Transaction cost considerations
- Liquidity constraints

## Configuration

### Model Configuration (`domain/config.py`)

```python
@dataclass
class ModelConfig:
    # Feature dimensions
    f_alpha: int = 900      # Number of alpha factors
    f_style: int = 40       # Number of style factors
    f_meta: int = 2         # Number of meta features
    hist_len: int = 20      # Historical lookback period
    
    # Model dimensions
    d_style: int = 32       # Style factor embedding
    d_alpha: int = 64       # Alpha factor embedding
    d_tmp: int = 32         # Temporal embedding
    d_model: int = 96       # Overall model dimension
    d_edge: int = 8         # Edge feature dimension
    
    # Graph topology
    topk_sim: int = 20      # Top-k similarity connections
    topk_dyn: int = 20      # Top-k dynamic connections
    topk_latent: int = 20   # Top-k latent connections
    
    # Architecture
    n_prop_layers: int = 2   # Number of propagation layers
    n_heads: int = 4        # Multi-head attention
    composer: str = "semiring"  # Composition method
    
    # Graph toggles
    use_prior: bool = True
    use_sim: bool = True
    use_dynamic: bool = True
    use_latent: bool = True
```

### Experiment Configuration

- **Data Paths**: Factor data, labels, and liquidity files
- **Training Parameters**: Learning rate, batch size, epochs
- **Evaluation Settings**: Portfolio size, capital allocation
- **Ablation Studies**: Systematic component removal experiments

## Usage

### Training a Single Season

```bash
python train.py --year 2023 --quarter 1 --ablation baseline --seed 42
```

### Running Full Ablation Study

```bash
python backtest.py
```

This evaluates all ablation configurations across multiple quarters and saves results to JSON.

### Ablation Configurations

The system supports comprehensive ablation studies:

- `baseline`: Full model with all components
- `no_latent`: Remove latent graph learner
- `no_prior`: Remove prior knowledge graph
- `no_dynamic`: Remove dynamic behavior graph
- `no_sim`: Remove factor similarity graph
- `sum_only`: Use sum composition instead of semiring
- `max_only`: Use max composition
- `agr_only`: Use aggregation composition
- `plain_attn`: Use standard attention
- `single_layer`: Reduce propagation layers

## Key Features

### 1. Multi-Relational Graph Learning
- Combines heterogeneous relationship types
- Flexible graph composition mechanisms
- Adaptive graph structure learning

### 2. Temporal Modeling
- Historical sequence encoding
- Dynamic correlation capture
- Season-aware training

### 3. Quantitative Finance Integration
- Factor-based investment framework
- Realistic portfolio simulation
- Industry-standard evaluation metrics

### 4. Extensible Architecture
- Modular component design
- Easy graph type addition
- Flexible composition methods

## Technical Implementation

### Dependencies
- PyTorch for deep learning
- PyTorch Lightning for training infrastructure
- Polars for efficient data processing
- NumPy for numerical operations

### Performance Considerations
- GPU-accelerated graph operations
- Efficient data loading with Polars
- Memory-conscious batch processing
- Checkpoint-based model saving

### Data Flow
1. Raw factor/label data loading
2. Preprocessing and standardization
3. Temporal splitting by quarters
4. Graph construction from features
5. Multi-relational message passing
6. Semiring composition
7. Ranking prediction
8. Portfolio evaluation

## Evaluation Metrics

### Information Coefficient (IC)
- Linear correlation between predicted scores and actual returns
- Measures ranking accuracy

### RankIC
- Spearman correlation for non-linear relationships
- More robust to outliers

### Portfolio Return
- Simulated top-k portfolio performance
- Realistic transaction costs and liquidity

### NDCG@200
- Normalized discounted cumulative gain
- Emphasizes top-ranking quality

### Stability
- Return consistency over time
- Risk-adjusted performance measure

## Research Contributions

This system advances quantitative finance by:

1. **Multi-Relational Modeling**: Captures diverse factor interactions beyond simple correlations
2. **Graph Neural Networks**: Applies modern GNN techniques to financial factor modeling
3. **Semiring Composition**: Novel mathematical framework for combining relational information
4. **Systematic Ablation**: Comprehensive understanding of component contributions
5. **Temporal Dynamics**: Time-varying relationship modeling in financial markets

## Future Directions

- Additional graph types (sector rotation, macro relationships)
- Advanced composition methods (attention-based, learning-based)
- Real-time adaptation mechanisms
- Cross-market applications
- Alternative data integration

## File Structure

```
Graph/
|-- cli/                    # Command-line interfaces
|-- data/                   # Data processing and loading
|-- domain/                 # Configuration and types
|-- evaluation/             # Metrics and portfolio simulation
|-- graphs/                 # Graph construction modules
|-- modeling/               # Model architecture components
|-- training/               # Training pipeline and utilities
|-- tests/                  # Unit tests
|-- train.py               # Training entry point
|-- backtest.py            # Evaluation entry point
```

This system represents a comprehensive approach to modern quantitative factor investing using graph neural networks and multi-relational learning.
