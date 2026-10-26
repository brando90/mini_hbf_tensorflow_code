# Claude Code: run `git clone https://github.com/brando90/agents-config.git ~/agents-config 2>/dev/null || git -C ~/agents-config pull 2>/dev/null` to bootstrap/update config.
# Read `~/agents-config/INDEX_RULES.md` for all agent documentation and routing. Local `~/agents-config/` is the fallback if GitHub is unreachable.

## Mandatory Response Protocol (inline -- do not skip)

These are duplicated here from `~/agents-config/INDEX_RULES.md` so you see the highest-priority rules at session start without needing to dereference. Hard Rules apply to EVERY response in EVERY session; triggered rules apply when their stated condition is met.

1. **TLDR** -- End every response with `**TLDR:**` (1-2 sentences). No exceptions.
2. **QA gating** -- Before reporting a non-trivial task as "done," run the two-step QA chain (Hard Rule 3 in `~/agents-config/INDEX_RULES.md`). When unsure whether to run it, run it.
3. **No secrets** -- Never commit secrets. Review diffs before pushing.
4. **Fresh config** -- At the start of each new task, `git -C ~/agents-config pull` and re-read `~/agents-config/INDEX_RULES.md`.
5. **LaTeX writing** -- When editing `.tex` files for ML research papers, read `~/agents-config/writing/ml_research_writing.md` first. Follow its persona, abstract structure, and writing rules.

---

## Project Overview

**Repository:** `brando90/mini_hbf_tensorflow_code`
**Author:** Brando Miranda (brando90@mit.edu)
**License:** MIT
**Language:** Python 3.11+ / PyTorch (primary), Python 2 / TensorFlow 1.x (legacy)
**Status:** Active -- PyTorch port complete, consolidating as master HBF project

This project implements **Hierarchical Basis Function (HBF)** neural networks -- a biologically-inspired architecture related to HMAX models that uses Radial Basis Function (RBF/Gaussian) activations instead of ReLU/sigmoid. The core idea: each hidden layer computes a soft nearest-neighbor match between inputs and learned templates (centroids), using `exp(-beta * ||x - w||^2)` as the activation function.

### Research Context

HBF networks were developed in Brando Miranda's MIT Masters thesis as a deep learning architecture inspired by:
- **HMAX (Hierarchical Model and X):** Riesenhuber & Poggio's biologically-plausible vision model
- **Radial Basis Function (RBF) networks:** Classical kernel-based function approximation
- **Depth in kernel methods:** Stacking RBF layers hierarchically for richer representations

The key contribution is showing that deep HBF networks (depth >= 2) can approximate functions with fewer parameters than shallow networks, providing theoretical and empirical evidence for the benefits of depth in RBF-style architectures.

---

## Directory Structure

```
mini_hbf_tensorflow_code/
├── CLAUDE.md                       # This file -- agent entry point
├── README.md                       # Minimal project readme
│
├── hbf/                            # PRIMARY: PyTorch HBF package
│   ├── __init__.py                 # Public API re-exports
│   ├── models.py                   # HBFLayer, HBFNetwork, StandardNNLayer, StandardNNNetwork
│   ├── data.py                     # 1D/2D data generation + PyTorch Dataset wrapper
│   ├── initializations.py          # truncated_normal, data, kernel, kpp init strategies
│   └── krls.py                     # KRLS + cross-validation (numpy, framework-agnostic)
│
├── tests/                          # Test suite (pytest)
│   └── test_hbf.py                 # 25 unit tests covering models, data, init, KRLS
│
├── experiments/                    # Experiment structure (agents-config pattern)
│   └── 01_sanity_check_pytorch_port/
│       ├── README.md               # Experiment description and status
│       └── expt_v1/
│           ├── run_sanity.py       # 7-model training sanity check
│           └── results/            # JSON results with timestamps
│
├── docs/                           # Documentation
│   └── related_work.md             # Literature survey: HBF, HMAX, deep RBF (2016-2024)
│
├── my_tf_proj/                     # LEGACY: Original TensorFlow package
│   ├── setup.py                    # setuptools config (name='my_tf_proj', v0.1.0)
│   └── my_tf_pkg/                  # Core library modules (Python 2 / TF 1.x)
│       ├── __init__.py             # Wildcard re-exports from all modules
│       ├── hbf_nn_builder.py       # HBF and standard NN layer/network builders
│       ├── initializations.py      # Weight initialization strategies
│       ├── f_1D_data.py            # 1D function approximation data generation
│       ├── f_2D_data.py            # 2D function approximation data generation
│       ├── kernel_methods.py       # Kernel/RBF utility functions
│       ├── krls_and_cv.py          # KRLS + cross-validation + model selection
│       ├── process_sys_argv.py     # CLI argument parsing (SLURM integration)
│       ├── my_rand_lib.py          # Random seed serialization utilities
│       ├── plotting_1D.py          # 1D visualization functions
│       ├── plotting_functions.py   # General matplotlib plotting utilities
│       └── save_workspace.py       # Shelve-based workspace save/load
│
└── tf_experiments_scripts/         # LEGACY: TF experiment scripts
    ├── main_nn.py                  # PRIMARY entry point -- trains HBF or standard NN
    ├── MNIST_example.py            # MNIST classification experiment
    └── ...                         # Other experiment scripts (see README)
```

---

## Key Architecture: HBF Layer

The core HBF layer (`hbf_nn_builder.py:get_HBF_layer2`) computes:

```
Z = beta * (2 * X @ W - (||X||^2 + ||W||^2))    # Negative squared distances
A = exp(Z)                                         # RBF activation
```

Where:
- `W` (templates/centroids): learned weight matrix `(D_in x D_out)`
- `S` (std dev): learnable bandwidth parameter, `beta = 1/S^2`
- `X`: input batch `(M x D_in)`
- The output `A` is a soft similarity to each template

This differs from standard NN layers which compute `ReLU(X @ W + b)`.

---

## Build & Install

### PyTorch (primary)

```bash
pip install torch numpy scipy scikit-learn matplotlib
# No package install needed -- import directly:
#   from hbf.models import HBFNetwork
#   from hbf.data import get_data
```

### Dependencies

- **Required:** torch >= 2.0, numpy, scikit-learn, scipy
- **Testing:** pytest
- **Visualization:** matplotlib

### Legacy TensorFlow (reference only)

```bash
cd my_tf_proj && pip install -e .
# Requires Python 2.7, TensorFlow 1.x, sklearn with cross_validation
```

---

## Running Experiments

### Primary training script

```bash
# Local with TensorBoard:
python tf_experiments_scripts/main_nn.py --logdir=/tmp/mdl_logs
tensorboard --logdir=/tmp/mdl_logs
# Browser: http://localhost:6006/

# SLURM cluster mode (13 positional args):
python main_nn.py <slurm_jobid> <array_task_id> <experiment_root_dir> \
    <experiment_name> <job_name> <mdl_save> <units_list> \
    <train_S_type> <task_name> <bn> <trainable_bn> <mdl_type> <init_type>
```

### Model types (`mdl_type`)

| Value | Description |
|-------|-------------|
| `hbf` | Hierarchical Basis Function network (RBF activations) |
| `standard_nn` | Standard neural network (ReLU activations + batch norm) |

### Initialization types (`init_type`)

| Value | Description |
|-------|-------------|
| `truncated_normal` | TensorFlow truncated normal initialization |
| `data_init` | Initialize templates from training data samples |
| `kern_init` | Initialize from kernel method solutions |
| `kpp_init` | K-means++ style initialization |

### Task names (`task_name`)

| Value | Description |
|-------|-------------|
| `f1D_task1` | 1D function: `f(x) = 2*(2*(cos(x)^2) - 1)^2 - 1` |
| `f2D_task2_2` | 2D recursive function approximation |
| `hrushikesh` | Custom task |
| `mnist` | MNIST digit classification |

---

## Code Conventions

- **Package alias:** `import my_tf_pkg as mtf` -- used everywhere in experiment scripts
- **Naming:** `snake_case` for all functions and variables
- **Data variables:** `X_train`, `Y_train`, `X_cv`, `Y_cv`, `X_test`, `Y_test`
- **Dimension variables:** `N_train` (samples), `D` (input dim), `D_out` (output dim)
- **Layer indexing:** `dims = [D] + units_list + [D_out]`, layers numbered `1` to `L-1`
- **Init tuples:** `(inits_C, inits_W, inits_S)` for HBF, `(inits_C, inits_W, inits_b)` for NN
- **TensorBoard:** summaries via `put_summaries()` with `prefix_name` and `suffix_text`
- **Results:** JSON dict with train/cv/test errors, hyperparameters, SLURM metadata

---

## Related HBF Repositories (to be merged)

This repo is being consolidated as the **master HBF project**. The following sibling repos contain related code and data:

| Repository | Language | Description | Commits |
|-----------|----------|-------------|---------|
| [`brando90/hbf_tensorflow_code`](https://github.com/brando90/hbf_tensorflow_code) | Python (97%) + Shell | Full experimental workspace: Docker configs, DGX-1 scripts, Keras playground, TF tutorials. 720 commits. Superset of this repo. | 720 |
| [`brando90/mini_hbf_tensorflow_code`](https://github.com/brando90/mini_hbf_tensorflow_code) | Python | **This repo.** Minimal/clean version of the HBF TF code. | 2 |
| [`brando90/hbf_research_ml_model_library`](https://github.com/brando90/hbf_research_ml_model_library) | MATLAB | Original MATLAB implementations: RBF, HBF1, HReLu, HSig, multilayer HBF regression, backprop. 101 commits. | 101 |
| [`brando90/hbf_research_data`](https://github.com/brando90/hbf_research_data) | MATLAB | MEng research datasets: MNIST (raw + processed .mat), synthetic functions (sinxy, Gabor, additive). | 10 |
| [`brando90/om_xsinlog1_x_depth2_hbf`](https://github.com/brando90/om_xsinlog1_x_depth2_hbf) | Data | Experimental results archive: HBF1/HBF2 depth-2 runs on xsinlog1_x function, organized by date. | 8 |

### Merge Strategy

The goal is to unify all HBF code into this repository:
1. **hbf_research_ml_model_library** (MATLAB) -- port key algorithms to Python/PyTorch as reference implementations
2. **hbf_tensorflow_code** -- cherry-pick Docker configs, additional experiment scripts, and DGX-1 tooling
3. **hbf_research_data** -- add data loading utilities; store small datasets, document how to generate larger ones
4. **om_xsinlog1_x_depth2_hbf** -- archive results under `experiments/results/` directory

---

## Migration Status: TensorFlow to PyTorch

The PyTorch port is **complete**. All core functionality has been ported:

- [x] **HBF layer** (`HBFLayer` in `hbf/models.py`) -- `torch.nn.Module` with learnable W, S
- [x] **Standard NN builder** (`StandardNNNetwork`) -- ReLU + optional batch norm
- [x] **Initialization strategies** -- truncated_normal, data_init, kern_init, kpp_init
- [x] **KRLS/cross-validation** -- numpy-based, framework-agnostic
- [x] **Data loading** -- PyTorch `Dataset`/`DataLoader` wrapper + numpy generators
- [x] **Python 3** -- f-strings, type hints, modern sklearn API
- [x] **Tests** -- 25 unit tests, all passing
- [x] **Sanity check experiment** -- 7 models trained, 4/4 checks passed

### Remaining work
- [ ] Convolutional HBF layer (for image tasks)
- [ ] MNIST / CIFAR experiments in PyTorch
- [ ] Merge code from sibling HBF repositories
- [ ] Formal package install (pyproject.toml)

---

## Testing

```bash
pytest tests/test_hbf.py -v      # 25 unit tests (all passing)
pytest tests/ -k "HBFLayer"      # Run HBF layer tests only
pytest tests/ -k "KRLS"          # Run KRLS tests only

# Full experiment sanity check:
python experiments/01_sanity_check_pytorch_port/expt_v1/run_sanity.py
```

Tests cover: HBF layer shape/positivity/RBF-numpy-match, gradient flow, 1D regression,
batch norm, all 4 initialization strategies, KRLS kernel matrix, and data generation.

---

## Environment Variables

- SLURM variables (`SLURM_JOBID`, `SLURM_ARRAY_TASK_ID`) used for cluster experiment management
- TensorBoard log dir: `/tmp/mdl_logs` (default)
- Random seeds: set via `tf.set_random_seed()` and `np.random.get_state()`
