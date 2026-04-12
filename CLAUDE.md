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
**Language:** Python 2 (legacy TensorFlow 1.x with `tf.contrib`)
**Status:** Research prototype from MIT MEng thesis (~2016) -- being consolidated as the master HBF project

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
├── my_tf_proj/                     # Installable Python package
│   ├── setup.py                    # setuptools config (name='my_tf_proj', v0.1.0)
│   └── my_tf_pkg/                  # Core library modules
│       ├── __init__.py             # Wildcard re-exports from all modules
│       ├── hbf_nn_builder.py       # HBF and standard NN layer/network builders
│       ├── initializations.py      # Weight initialization strategies (data_init, kern_init, kpp_init)
│       ├── f_1D_data.py            # 1D function approximation data generation
│       ├── f_2D_data.py            # 2D function approximation data generation
│       ├── kernel_methods.py       # Kernel/RBF utility functions
│       ├── krls_and_cv.py          # KRLS + cross-validation + model selection
│       ├── process_sys_argv.py     # CLI argument parsing (SLURM integration)
│       ├── my_rand_lib.py          # Random seed serialization utilities
│       ├── plotting_1D.py          # 1D visualization functions
│       ├── plotting_functions.py   # General matplotlib plotting utilities
│       └── save_workspace.py       # Shelve-based workspace save/load
└── tf_experiments_scripts/         # Runnable experiment scripts
    ├── main_nn.py                  # PRIMARY entry point -- trains HBF or standard NN
    ├── MNIST_example.py            # MNIST classification experiment
    ├── krls.py                     # KRLS baseline experiment
    ├── f2D_krls.py                 # 2D function fitting with KRLS
    ├── f2D_krls_random.py          # 2D KRLS with random hyperparameters
    ├── f2D_recursive_krls.py       # Recursive KRLS on 2D functions
    ├── f2D_plot_original_func.py   # Visualization of target 2D functions
    ├── generate_2D_data.py         # Data generation script
    ├── get_sense_norm.py           # Norm analysis utilities
    ├── get_stuff.py                # Misc extraction script
    ├── krls_collect_data.py        # KRLS data collection
    ├── krls_plot_errrors_script.py  # Error curve plotting
    └── multiple_vs_single_collect_results.py  # Multi-model comparison
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

```bash
cd my_tf_proj
pip install -e .    # Development install
# or
python setup.py install
```

### Dependencies (implicit -- not all in setup.py)

- **Required:** numpy >= 1.11.0, tensorflow (1.x with contrib), scikit-learn, scipy
- **Visualization:** matplotlib (with mpl_toolkits.mplot3d)
- **System:** Python 2.7 (uses `print` statements, `xrange`, `encode('hex')`)

### Known Compatibility Issues

- `tensorflow.contrib.*` was removed in TensorFlow 2.x
- `sklearn.cross_validation` was deprecated (now `sklearn.model_selection`)
- Python 2 print statements (`print 'foo'`) need parentheses for Python 3
- `xrange` does not exist in Python 3

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

## Migration Roadmap: TensorFlow to PyTorch

This codebase currently uses legacy TensorFlow 1.x (Python 2). The planned migration path:

1. **Port core HBF layer** (`get_HBF_layer2`) to `torch.nn.Module`
2. **Port standard NN builder** to PyTorch (straightforward -- uses standard ops)
3. **Port initialization strategies** (data_init, kern_init, kpp_init) to work with PyTorch tensors
4. **Port KRLS/cross-validation** utilities (numpy-based, mostly framework-agnostic)
5. **Update data loading** to use PyTorch `Dataset`/`DataLoader`
6. **Modernize Python** to Python 3.8+ (print functions, type hints, f-strings)
7. **Add tests** (currently no test suite exists)

---

## Testing

**Current state:** No test suite exists. Experiment scripts in `tf_experiments_scripts/` serve as integration tests.

**Planned:**
```bash
# After PyTorch migration:
pytest tests/           # Unit tests
pytest tests/ -k hbf    # HBF-specific tests
```

---

## Environment Variables

- SLURM variables (`SLURM_JOBID`, `SLURM_ARRAY_TASK_ID`) used for cluster experiment management
- TensorBoard log dir: `/tmp/mdl_logs` (default)
- Random seeds: set via `tf.set_random_seed()` and `np.random.get_state()`
