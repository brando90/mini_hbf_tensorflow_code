# 01 - Sanity Check: PyTorch Port

## Goal
Verify that the PyTorch port of HBF networks reproduces correct behavior:
1. HBF layer computes RBF activations correctly (matches numpy reference)
2. HBF network trains and reduces loss on 1D function approximation
3. Standard NN network trains and reduces loss on same task
4. KRLS baseline produces reasonable fits
5. All initialization strategies work without errors

## Structure
```
expt_v1/
├── run_sanity.py           # Main training + evaluation script
└── results/                # Output logs and plots
```

## Method
- Train HBF and standard NN on f(x) = 2*(2*cos(x)^2 - 1)^2 - 1 (1D task)
- Compare final MSE against KRLS baseline
- Verify loss decreases monotonically (on train set)

## Dependencies
- PyTorch, numpy, scikit-learn, scipy

## Status

| Version | Status | Notes |
|---------|--------|-------|
| v1 | In progress | Initial sanity check |
