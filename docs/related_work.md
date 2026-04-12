# Related Work: HBF, HMAX, and Deep RBF Networks

A survey of work related to Hierarchical Basis Function (HBF) neural networks, covering developments since Brando Miranda's MIT MEng thesis (2016).

---

## 1. Foundational Work

### 1.1 HBF Networks (Miranda, 2016)

- **"Training Hierarchical Networks for Function Approximation"**
  Brando Miranda, MIT MEng Thesis, 2016. Supervisor: Prof. Tomaso Poggio.
  [MIT DSpace](https://dspace.mit.edu/handle/1721.1/113159)

  Introduced HBF networks: deep architectures with stacked RBF layers using learnable Gaussian templates. Each layer computes `A = exp(-beta * ||x - w||^2)`, generalizing the HMAX model with end-to-end trainable parameters. Demonstrated that deep HBF (depth >= 2) can approximate compositional functions more efficiently than shallow networks.

### 1.2 Why and When Deep Networks Avoid the Curse of Dimensionality (Poggio et al., 2017)

- **"Why and when can deep-but not shallow-networks avoid the curse of dimensionality: A review"**
  Tomaso Poggio, Hrushikesh Mhaskar, Lorenzo Rosasco, Brando Miranda, Qianli Liao.
  International Journal of Automation and Computing (IJAC), 2017. **Most Cited Paper Certificate.**
  [Springer](https://link.springer.com/article/10.1007/s11633-017-1054-2) | [Semantic Scholar](https://www.semanticscholar.org/paper/fdbe123a6535720274db2518af49abe9e9daac87)

  Proved that for compositional functions (expressible as a DAG of simpler functions), deep networks avoid the curse of dimensionality while shallow networks cannot. The compositionality structure allows deep networks to exploit the same DAG structure, achieving exponentially better approximation rates. This provides the theoretical foundation for why depth matters in HBF networks.

### 1.3 HMAX (Riesenhuber & Poggio, 1999)

- **"Hierarchical models of object recognition in cortex"**
  Maximilian Riesenhuber, Tomaso Poggio. Nature Neuroscience, 1999.

  The biological inspiration for HBF. HMAX models the ventral visual cortex with alternating simple cells (template matching via Gaussian tuning) and complex cells (max-pooling for invariance). HBF generalizes HMAX by making all templates and bandwidths learnable via gradient descent.

---

## 2. Approximation Theory for Deep Networks (2017-2023)

### 2.1 Extensions of Poggio-Mhaskar Theory

- **"How Deep Sparse Networks Avoid the Curse of Dimensionality: Efficiently Computable Functions are Compositionally Sparse"**
  Tomaso Poggio, Hrushikesh Mhaskar. CBMM Memo No. 138, 2023.
  [CBMM](https://cbmm.mit.edu/publications/how-deep-sparse-networks-avoid-curse-dimensionality-efficiently-computable-functions) | [NSF](https://par.nsf.gov/biblio/10565465)

  Extends the 2017 results by showing that sparsity of compositional target functions naturally leads to sparse deep networks. Demonstrates that efficiently computable functions are compositionally sparse, providing a stronger characterization of when depth helps.

- **"Function Approximation by Deep Networks"**
  Hrushikesh Mhaskar. Communications on Pure and Applied Analysis, 2020.
  [arXiv:1905.12882](https://arxiv.org/abs/1905.12882) | [AIMS](https://www.aimsciences.org//article/doi/10.3934/cpaa.2020181)

  Rigorous analysis of function approximation capabilities of deep networks, extending the compositional approximation framework.

- **"Theoretical Issues in Deep Networks"**
  Tomaso Poggio et al. PNAS, 2020.
  [PNAS](https://www.pnas.org/doi/10.1073/pnas.1907369117)

  Comprehensive treatment of approximation, optimization, and generalization theory for deep networks, unifying results on compositionality with practical network architectures.

### 2.2 Generalization Theory

- **"Theory IIIb: Generalization in Deep Networks"**
  Tomaso Poggio, Qianli Liao, Brando Miranda et al. arXiv:1806.11379, 2018.
  [arXiv](https://arxiv.org/abs/1806.11379)

  Explores generalization bounds for deep networks, including the role of compositionality and norm-based complexity measures.

- **"Theory III: Dynamics and Generalization in Deep Networks"**
  Tomaso Poggio et al. arXiv:1903.04991, 2019.
  [arXiv](https://arxiv.org/abs/1903.04991)

  Studies the dynamics of SGD and its implicit regularization effects on generalization.

---

## 3. HMAX and Biologically-Inspired Models (2014-2024)

### 3.1 CHMAX: Modernized HMAX (2024)

- **"HMAX Strikes Back: Self-supervised Learning of Human-Like Scale Invariant Representations"**
  CCN (Cognitive Computational Neuroscience) 2024.
  [PDF](https://2024.ccneuro.org/pdf/533_Paper_authored_CCN_2024_HMAX-(2).pdf)

  Augments the original HMAX hierarchy with trainable filters and self-supervised learning objectives. Achieves human-like scale-invariant generalization while preserving the biological plausibility of the original architecture. Demonstrates that HMAX-style architectures can compete with modern DNNs when properly trained. **Directly relevant to HBF** -- both approaches make HMAX-style architectures trainable, but CHMAX focuses on self-supervised learning while HBF focuses on supervised function approximation.

### 3.2 Enhanced HMAX Variants

- **"Enhanced HMAX model with feedforward feature learning for multiclass categorization"**
  PubMed, 2015. [PubMed](https://pubmed.ncbi.nlm.nih.gov/26500532/)

  Extends HMAX with learnable features in the feedforward path.

- **"C-HMAX: Artificial Cognitive Model Inspired by the Color Vision Mechanism of the Human Brain"**
  IEEE Xplore, 2013. [IEEE](https://ieeexplore.ieee.org/document/6449407/)

  Extends HMAX to handle color information, modeling color-sensitive neurons.

- **"Biological modeling of human visual system for object recognition using GLoP filters and sparse coding on multi-manifolds"**
  Machine Vision and Applications, 2018.
  [Springer](https://link.springer.com/article/10.1007/s00138-018-0928-9)

  Combines biologically-inspired Gabor-like filters with sparse coding on the HMAX framework.

### 3.3 Other Biologically-Inspired Models

- **CORnet** (DiCarlo lab, MIT): Recurrent neural network models of the ventral visual stream. Uses standard convolutions rather than RBF activations.

- **Capsule Networks** (Hinton et al., 2017): Template-based hierarchical models using dynamic routing. Conceptually related to HBF in using "template matching" rather than convolution, but with different mechanics (agreement-based routing vs. RBF similarity).

---

## 4. Kernel Methods Meet Deep Learning

### 4.1 Deep Kernel Learning

- **"Deep Kernel Learning"**
  Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, Eric P. Xing.
  AISTATS 2016.
  [arXiv:1511.02222](https://arxiv.org/abs/1511.02222) | [PDF](http://proceedings.mlr.press/v51/wilson16.pdf)

  Combines deep neural network feature extraction with Gaussian process (GP) kernel methods. Transforms inputs through a deep architecture before applying a spectral mixture base kernel. Achieves O(n) training and O(1) prediction. **Relation to HBF**: DKL uses a deep network as a feature extractor feeding into a GP, while HBF uses RBF kernels directly as the activation function at every layer -- a more tightly integrated approach.

### 4.2 Neural Tangent Kernel (NTK)

- **"Neural Tangent Kernel: Convergence and Generalization in Neural Networks"**
  Arthur Jacot, Franck Gabriel, Clement Hongler. NeurIPS 2018.
  [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)

  Shows that infinitely wide neural networks are equivalent to kernel regression with the NTK. The NTK of an RBF-activation network (like HBF) would have interesting connections to compositions of Gaussian kernels. **Gap**: The NTK analysis of HBF-style networks (with exp activations instead of ReLU) has not been thoroughly studied.

- **"Deep Neural Networks as Gaussian Processes"**
  Jaehoon Lee et al. ICLR 2018.
  [OpenReview](https://openreview.net/pdf?id=B1EA-M-0Z)

  Shows that deep networks with i.i.d. weights converge to GPs in the infinite-width limit. For HBF networks, the corresponding GP kernel would be a hierarchical composition of RBF kernels.

### 4.3 Random Features for Kernels

- **"Random Features for the Neural Tangent Kernel"**
  2021. [ResearchGate](https://www.researchgate.net/publication/350646972_Random_Features_for_the_Neural_Tangent_Kernel)

  Develops random feature approximations for NTKs, connecting to the random features framework (Rahimi & Recht, 2009). Relevant to scaling HBF-style networks via random feature approximations.

---

## 5. Deep RBF Networks and Implementations

### 5.1 Existing PyTorch RBF Implementations

- **RBF-Pytorch** ([GitHub](https://github.com/zxdclyz/RBF-Pytorch)): Single-layer Gaussian kernel RBF. No depth support.
- **PytorchRBFLayer** ([GitHub](https://github.com/rssalessio/PytorchRBFLayer)): Customizable RBF layer with multiple basis function options. Single-layer only.
- **GPyTorch RBFKernel** ([Docs](https://docs.gpytorch.ai/en/stable/kernels.html)): RBF kernel for Gaussian processes, not for deep stacking.

**Key gap**: All existing PyTorch RBF implementations are single-layer. Our HBF implementation is (to our knowledge) the only open-source deep/hierarchical RBF network in PyTorch with multiple stacked layers and principled initialization strategies.

### 5.2 Keras/TensorFlow RBF

- **Kaggle: Radial Basis Networks and Custom Keras Layers** ([Kaggle](https://www.kaggle.com/code/residentmario/radial-basis-networks-and-custom-keras-layers)): Tutorial implementing single-layer RBF in Keras.

---

## 6. Brando Miranda's Subsequent Work (2018-2024)

- **Stanford PhD** (admitted Autumn 2022, advisor: Prof. Sanmi Koyejo).
  [Stanford Profile](https://profiles.stanford.edu/brando-miranda) | [Google Scholar](https://scholar.google.com/citations?user=_NQJoBkAAAAJ&hl=en) | [Personal site](https://brando90.github.io/brandomiranda/)

- **"The Curse of Low Task Diversity: On the Failure of Transfer Learning to Outperform MAML and Their Empirical Equivalence"** (2022). Co-authored with Patrick Yu, Yu-Xiong Wang, Sanmi Koyejo. Studies meta-learning and transfer learning trade-offs.

- **NeurIPS 2023 Outstanding Main Track Paper Award** (top 0.4%, 2 papers selected). Research on data-centric ML for frontier models.

- **ICML 2024 Outstanding Paper** at TiFA Workshop. Research on trustworthy foundation models.

- Research interests have expanded to meta-learning, foundation models for theorem proving, LLM pre-training, and human/brain-inspired AI.

---

## 7. Gaps and Opportunities

Based on this survey, the following gaps present opportunities for HBF research:

1. **No deep RBF in PyTorch**: All existing open-source RBF implementations are single-layer. Our HBF package fills this gap with stackable layers and principled initialization.

2. **NTK analysis of HBF**: The Neural Tangent Kernel of networks with `exp(-||x-w||^2)` activations (rather than ReLU) has not been characterized. This could provide new generalization bounds.

3. **CHMAX + HBF synergy**: CHMAX (2024) shows that trainable HMAX can match DNNs with self-supervised learning. Combining CHMAX-style training with HBF's flexible depth and bandwidth learning could yield a powerful biologically-plausible architecture.

4. **HBF for modern tasks**: The original HBF was demonstrated on 1D/2D function approximation and MNIST. Scaling to modern vision tasks (CIFAR, ImageNet) with convolutional HBF layers is unexplored.

5. **Mixture of Experts connection**: HBF's soft template matching (Gaussian gating) is conceptually similar to MoE soft routing. Exploring this connection could lead to HBF-style expert selection.

6. **Foundation model kernels**: Softmax attention in transformers can be viewed as a kernel operation. HBF's RBF kernels could potentially serve as attention replacements with better-understood approximation properties.

7. **Approximation theory for HBF depth > 2**: The Poggio-Mhaskar theory proves benefits of depth for compositional functions, but tight bounds specific to HBF (with RBF activations) at depths > 2 remain open.
