export H_ens, EKRMLEObj, EKRMLE_run!, EKRMLE_step!


@inline _colmean(V::AbstractMatrix) = vec(mean(V, dims=2)) # computes sample mean (column-wise)

function _samplecov(V::AbstractMatrix)
    # computes sample covariance (column-wise)
    J = size(V, 2)
    μ = _colmean(V)
    X = V .- μ
    return (X * X') / (J-1)
end

function _samplecrosscov(V1::AbstractMatrix, V2::AbstractMatrix)
    # computes sample cross-covariance between two ensembles with common ensemble size J
    J = size(V1, 2)
    μ₁ = _colmean(V1)
    μ₂ = _colmean(V2)
    V1c = V1 .- μ₁
    V2c = V2 .- μ₂
    return (V1c * V2c') / (J-1)
end

"""
    H_ens(params, V, H) -> O

Propagates all ensemble members (columns) of `V` through forward map ``H``

# Arguments
- `params` : arbitrary problem parameters
- `V::AbstractMatrix{T}` : matrix of ensemble members, size `(d × J)`
    - `V[:, j]` is the j-th ensemble member ``v^{(j)}``
    - `d` is the state dimension, `J` is the ensemble size 
- `H::Function` : forward model; accepts `(params, v::AbstractVector)` and returns vector of length `n`

# Returns
- `O::Matrix{T}` : matrix of forward model evaluations, size `(n × J)`
    - `O[:, j] = H(params, V[:, j])`
    - `n` is the dimension of the observation space

# Notes
- Works for both **linear and nonlinear** `H`
- Calls to `H` are distributed over threads using `Threads.@threads`.
  Ensure `H` is **thread-safe** (no shared mutable state).
- Uses `view(V, :, j)` to avoid unnecessary copies when accessing columns.


"""

function H_ens(params, V::AbstractMatrix{T}, H::Function) where {T<:AbstractFloat}
    J = size(V, 2)
    h₁ = H(params, view(V, :, 1))
    n = length(h₁)
    O = Matrix{T}(undef, n, J)
    O[:, 1] .= h₁
    Threads.@threads for j in 2:J
        O[:, j] = H(params, view(V, :, j))
    end
    return O
end

"""
    EKRMLEObj{T,TI}

State container for EKRMLE with column-wise ensembles.

# Fields
- `V::Vector{Matrix{T}}` :
    History of ensembles over iterations. Each `V[i]` has size `(d × J)`;
    column `V[i][:, j]` is particle `v^{(j)}` at iteration `i-1`.
- `y::Vector{T}` :
    Observation vector `y ∈ ℝ^n`.
- `Γ::Matrix{T}` :
    Observation noise covariance `Γ ∈ ℝ^{n×n}` (symmetric PSD).
- `Yrand::Matrix{T}` :
    Matrix `(n × J)` with fixed per-particle perturbed observations
    `y^{(j)} = y + ε^{(j)}`, sampled once and reused each iteration.
- `Ohist::Vector{Matrix{T}}` :
    History of forward evaluations; each `(n × J)` stores `H(V[i])` for the
    **pre-update** ensemble at that iteration.
- `J::TI` :
    Ensemble size.
- `d::TI` :
    Parameter dimension.
- `n::TI` :
    Observation dimension.
- `iters::TI` :
    Number of update steps performed so far.

# Conventions
- Ensembles are column-wise: particles are the **columns** of `V[k]`.
- `Ytilde` uses the *same* `ε^{(j)}` across all iterations (Algorithm 2).
"""


mutable struct EKRMLEObj{T<:AbstractFloat, TI<:Int}
    V::Vector{Matrix{T}}
    y::Vector{T}
    Γ::Matrix{T}
    Yrand::Matrix{T}
    Ohist::Vector{Matrix{T}}
    J::TI; d::TI; n::TI
    iters::TI
end

function EKRMLEObj(
    V₀::AbstractMatrix{T}, # (d × J) columns are v₀^{(j)}
    y::AbstractVector{T}, # length n
    Γ::AbstractMatrix{T}; # (n × n)
    rng::AbstractRNG=Random.default_rng(),
) where {T<:AbstractFloat}
    d, J = size(V₀)
    n = length(y)
    𝛆 = MvNormal(Symmetric(Matrix{T}(Γ)))
    Yrand = Matrix{T}(undef, n, J)
    @inbounds for j = 1:J
        Yrand[:, j] = y .+ rand(rng, 𝛆)
    end
    EKRMLEObj{T,Int}(
        [Matrix{T}(V₀)],
        Vector{T}(y),
        Matrix{T}(Γ),
        Yrand,
        Matrix{T}[],
        J, d, n,
        0
    )
end

"""
    EKRMLE_step!(obj, Hens; rng=Random.default_rng()) -> nothing

Perform **one** EKRMLE iteration (Algorithm 2, column-wise ensembles).

# Arguments
- `obj::EKRMLEObj` :
    State container holding the current ensemble `V[end] :: (d × J)`,
    observation `y :: (n)`, noise covariance `Γ :: (n × n)`, fixed perturbed
    data `Yrand :: (n × J)`, history arrays, sizes, and params.
- `Hens::Function` :
    Batched forward evaluator. For an ensemble `V :: (d × J)`, it must return
    `O :: (n × J)` with `O[:, j] = H(s_param, V[:, j])`.

"""

function EKRMLE_step!(obj::EKRMLEObj{T}, Hens::Function) where {T<:AbstractFloat}
    # Current ensemble and forward evaluation
    V = obj.V[end]
    O = Hens(V)
    
    # Covariances
    Chh = _samplecov(O)
    Cvh = _samplecrosscov(V, O)

    # Compute Kalman gain
    A = Chh + obj.Γ
    AL = cholesky(A; check=false)
    n = size(O, 1)
    #K = Cvh*(AL \ I(n))
    K = Cvh/A

    # Residuals
    Θ = obj.Yrand .- O

    # Update ensemble
    Vnew = similar(V)
    @inbounds for j = 1:obj.J
        Vnew[:, j] = view(V, :, j) .+ K * view(Θ, :, j)
    end

    # push changes
    push!(obj.V, Vnew)
    push!(obj.Ohist, O)
    obj.iters += 1

    return nothing

end


function EKRMLE_step_stoch!(obj::EKRMLEObj{T}, Hens::Function) where {T<:AbstractFloat}
    # Current ensemble and forward evaluation
    V = obj.V[end]
    O = Hens(V)
    
    # Covariances
    Chh = _samplecov(O)
    Cvh = _samplecrosscov(V, O)

    # Compute Kalman gain
    A = Chh + obj.Γ
    AL = cholesky(A; check=false)
    n = size(O, 1)
    #K = Cvh*(AL \ I(n))
    K = Cvh/A

    # Residuals
    Θ = obj.Yrand .- O

    # Fresh Gaussian noise
    J = size(Θ, 2)
    𝛆 = MvNormal(Symmetric(Matrix{T}(obj.Γ)))
    @inbounds for j = 1:J
        Θ[:, j] .+= rand(𝛆)
    end

    # Update ensemble
    Vnew = similar(V)
    @inbounds for j = 1:obj.J
        Vnew[:, j] = view(V, :, j) .+ K * view(Θ, :, j)
    end

    # push changes
    push!(obj.V, Vnew)
    push!(obj.Ohist, O)
    obj.iters += 1

    return nothing

end


"""
    EKRMLE_run!(obj, params, H, N_iters) -> EKRMLEObj

Run EKRMLE (Algorithm 2) for `N_iters` iterations **in place** on `obj`.

# Arguments
- `obj::EKRMLEObj` :
    Initialized state (must contain `V[end]` as the current `(d × J)` ensemble,
    `y`, `Γ`, `Yrand`, sizes).
- `params` :
    Problem/model parameters to pass to the single-particle forward `H`.
    Can be any type (e.g., struct, NamedTuple).
- `H::Function` :
    Single-particle forward map. Must accept `(params, v::AbstractVector)`
    and return a vector of length `n` (the observation dimension).
- `N_iters::Integer` :
    Number of EKRMLE iterations to perform.

# Behavior
- Internally constructs a **batched** forward evaluator
  `Hens(V) = H_ens(params, V, H)` that maps `(d × J) → (n × J)`.
- Repeats `N_iters` times:
    1. `O = Hens(V)` to evaluate the forward map on all columns.
    2. `EKRMLE_step!(obj, Hens)` to compute the gain and update `V`.

# Returns
- The same `obj` (mutated), with:
  - `obj.iters` incremented by `N_iter`,
  - `N_iters` new ensembles appended to `obj.V`,
  - `N_iters` new forward outputs appended to `obj.Ohist`.

# Notes
- `H` may be **linear or nonlinear**; no derivatives required.
- Threading in `H_ens` is used if available; ensure `H` is thread-safe.
- If you prefer, you can pass a pre-built batched function instead of `H`
  by calling `EKRMLE_step!` directly in your own loop.
"""

function EKRMLE_run!(obj::EKRMLEObj{T}, params, H::Function, N_iters::TI) where {T<:AbstractFloat, TI<:Int}
    Hens = V -> H_ens(params, V, H)
    for _ in 1:N_iters
        EKRMLE_step!(obj, Hens)
        #EKRMLE_step_stoch!(obj, Hens)
    end
    return obj
end


