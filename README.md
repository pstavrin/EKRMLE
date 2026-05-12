# **E**nsemble **K**alman **R**andomized **M**aximum **L**ikelihood **E**stimation

Julia code that implements the EKRMLE algorithm, and reproduces all numerical experiments from [this paper](https://arxiv.org/abs/2507.03207 "Original EKRMLE paper").

## Installation

### Clone repository
```bash
git clone https://github.com/pstavrin/EKRMLE.git
cd EKRMLE
```

### Start Julia in project directory
```bash
julia
```

### Activate environment
```julia
using Pkg
Pkg.activate(".")
```

### Install dependencies
```julia
Pkg.instantiate()
```

## Contents

### Algorithms
* `EKRMLEHelpers.jl`: EKRMLE algorithm
* `EKI.jl`: classical EKI algorithm (determinisitc & stochastic)

### Scripts
* `LinearConvergence.jl`: EKRMLE applied to a random linear problem with illustration of convergence in the appropriate convergent subspaces.
* `HeatEq2D.jl`: EKRMLE applied to a Bayesian inverse problem arising from a 2D heat equation. BT-accelerated EKRMLE also illustrated here.
* `HeatEq2DBT.jl`: experimental routine for study of EKRMLE applied to a 2D heat equation Bayesian inverse problem with varying ensemble sizes and reduced model ranks.
* `Darcy2D.jl`: EKRMLE applied to a nonlinear Bayesian inverse problem arising from Darcy flow. Comparisons with EKI (deterministic & stochastic) also shown here.
* `plotBT.jl`: plots BT results and recreates all relevant figures.