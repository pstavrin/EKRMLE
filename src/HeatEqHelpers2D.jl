export Heat2DParams, build_heat2d_params
export sample_prior_ic,
       integrate_full_order,
       integrate_full_order_at_indices,
       integrate_full_order_at_times,
       state_to_grid
export observation_grid_indices,
       observation_linear_indices,
       observation_matrix_2d,
       top_left_observation_indices,
       top_left_observation_matrix,
       two_block_observation_indices,
       two_block_observation_matrix,
       observation_time_indices,
       observe_trajectory,
       observe_initial_condition,
       add_gaussian_noise,
       build_synthetic_data
export build_explicit_forward_operator,
       build_RLS_data,
       build_explicit_HRLS_operator,
       build_implicit_forward_operator,
       build_implicit_HRLS_operator
export build_reduction_factors,
       build_reduced_operators,
       ReducedHeatOperators
export build_reduced_explicit_forward_operator,
       build_reduced_implicit_forward_operator,
       build_reduced_HRLS_operator,
       build_reduced_solver
export weighted_relative_error, weighted_norm




mutable struct Heat2DParams{T<:AbstractFloat, TI<:Int, MTA<:AbstractMatrix{T}, MTB<:AbstractMatrix{T}}
    # Geometry / discretization
    Ω::Tuple{Tuple{T,T},Tuple{T,T}}
    Nx::TI
    Ny::TI
    Δx::T
    Δy::T
    Δt::T
    T_stop::T
    tspan::Vector{T}

    # PDE / model settings
    diffusion_coeffs::T
    BC::Tuple{Symbol,Symbol}

    # Full-order operators
    A::MTA
    B::MTB

    # Dimensions
    d::TI
    time_dim::TI

    # Boundary control input over time
    Ubc::Matrix{T}

    # Prior information
    Γpr_unmod::Matrix{T}
    Γpr_R::Matrix{T}
    Γpr::Matrix{T}

    # Fast integrator object
    solver
end


function build_heat2d_params(;
    Ω=((0.0, 1.0), (0.0, 1.0)),
    Nx::Int=40,
    Ny::Int=40,
    Δt::Float64=1e-2,
    T_stop::Float64=2.0,
    diffusion_coeffs::Float64=0.5,
    BC::Tuple{Symbol,Symbol}=(:dirichlet, :dirichlet),
    τ::Float64=3/4,
    α::Float64=10.0,
    Ubc_val::AbstractVector{<:Real}=[0.0, 0.0, 0.0, 0.0],
)
    heat2d = Heat2DModel(
        spatial_domain=Ω,
        time_domain=(zero(T_stop), T_stop),
        Δx=(Ω[1][2] + 1/Nx)/Nx,
        Δy=(Ω[2][2] + 1/Ny)/Ny,
        Δt=Δt,
        diffusion_coeffs=diffusion_coeffs,
        BC=BC,
    )

    A, B = heat2d.finite_diff_model(heat2d, heat2d.diffusion_coeffs)
    d = size(A, 2)
    time_dim = heat2d.time_dim

    Ubc = repeat(reshape(Float64.(Ubc_val), :, 1), 1, time_dim)

    Γpr_unmod = prior_covariance_2d(Nx, Ny, 1.0, 1.0, τ, α)
    Γpr_R = minmod_prior_compat(Γpr_unmod, Matrix(A))
    Γpr = Γpr_R' * Γpr_R

    solver = build_fast_be_solver(heat2d, diffusion_coeffs)

    return Heat2DParams(
        Ω,
        Nx,
        Ny,
        heat2d.Δx,
        heat2d.Δy,
        heat2d.Δt,
        T_stop,
        collect(heat2d.tspan),
        diffusion_coeffs,
        BC,
        A,
        B,
        d,
        time_dim,
        Ubc,
        Γpr_unmod,
        Γpr_R,
        Γpr,
        solver,
    )
end


function neumann_laplacian_1d_sparse(n::Int, h::Real)
    L = spzeros(Float64, n, n)

    for i in 2:n-1
        L[i, i-1] = 1.0
        L[i, i]   = -2.0
        L[i, i+1] = 1.0
    end

    L[1,1] = -1.0
    L[1,2] =  1.0
    L[n,n-1] =  1.0
    L[n,n]   = -1.0

    return L / h^2
end

function neumann_laplacian_2d(nx::Int, ny::Int, hx::Real, hy::Real)
    Lx = neumann_laplacian_1d_sparse(nx, hx)
    Ly = neumann_laplacian_1d_sparse(ny, hy)

    Ix = sparse(I, nx, nx)
    Iy = sparse(I, ny, ny)

    return kron(Iy, Lx) + kron(Ly, Ix)
end

function prior_covariance_2d(nx::Int, ny::Int, hx::Real, hy::Real, τ::Real, d::Real)
    Δ = neumann_laplacian_2d(nx, ny, hx, hy)
    N = nx * ny
    A = -Matrix(Δ) + τ^2 * I(N)

    F = eigen(Symmetric(A))
    λ = F.values
    Q = F.vectors

    return Q * Diagonal(λ .^ (-d)) * Q'
end

function minmod_prior_compat(Gamma0::AbstractMatrix, A::AbstractMatrix)
    R0 = cholesky(Symmetric(Gamma0)).U

    M0 = Symmetric(A * Gamma0 + Gamma0 * A')

    F = eigen(M0)
    eigval = F.values
    V = F.vectors

    flagPos = eigval .> 0
    if !any(flagPos)
        return R0
    end

    d_pos = sqrt.(eigval[flagPos])
    V_pos = V[:, flagPos]
    B = V_pos * Diagonal(d_pos)

    E = plyapc(A, B)'
    Fq = qr(vcat(R0, E))

    return Matrix(Fq.R)
end

"""
    sample_prior_ic(heat::Heat2DParams{T}; mean=nothing, rng=Random.default_rng())

Sample an initial condition from the Gaussian prior associated with `heat`.

By default, samples from N(0, Γpr). If `mean` is provided, samples from
N(mean, Γpr).
"""
function sample_prior_ic(
    heat::Heat2DParams{T};
    mean::Union{Nothing,AbstractVector{T}}=nothing,
    rng=Random.default_rng(),
) where {T<:AbstractFloat}

    μ = isnothing(mean) ? zeros(T, heat.d) : Vector{T}(mean)
    @assert length(μ) == heat.d "Mean vector must have length heat.d"

    return rand(rng, MvNormal(μ, Symmetric(heat.Γpr)))
end

"""
    integrate_full_order(heat::Heat2DParams, u0::AbstractVector)

Integrate the 2D heat equation from initial condition `u0` over the full
time grid `heat.tspan` using the precomputed fast backward Euler solver.

Returns a matrix `U` of size (d, nt), where each column is the state at
one time in `heat.tspan`.
"""
function integrate_full_order(
    heat::Heat2DParams,
    u0::AbstractVector,
)
    @assert length(u0) == heat.d "Initial condition must have length heat.d"

    U = integrate_model_fast(
        heat.solver,
        heat.B,
        heat.Ubc,
        heat.tspan,
        u0,
    )

    return U
end

"""
    integrate_full_order_at_indices(heat::Heat2DParams, u0::AbstractVector, idx)

Integrate from `u0` and return only the state snapshots indexed by `idx`
in the full time grid.

Returns a matrix of size (d, length(idx)).
"""
function integrate_full_order_at_indices(
    heat::Heat2DParams,
    u0::AbstractVector,
    idx::AbstractVector{<:Integer},
)
    @assert all(1 .<= idx .<= length(heat.tspan)) "Time indices out of bounds"

    U = integrate_full_order(heat, u0)
    return U[:, idx]
end

"""
    integrate_full_order_at_times(heat::Heat2DParams, u0::AbstractVector, times; nearest=true)

Integrate from `u0` and return state snapshots at the requested physical
times.

Currently this matches each requested time to the nearest point in
`heat.tspan`.

Returns `(Usel, idx)` where:
- `Usel` is the matrix of selected snapshots, size (d, length(times))
- `idx` are the corresponding indices in `heat.tspan`
"""
function integrate_full_order_at_times(
    heat::Heat2DParams{T},
    u0::AbstractVector,
    times::AbstractVector{T};
    nearest::Bool=true,
) where {T<:AbstractFloat}

    if !nearest
        error("Only nearest=true is currently implemented.")
    end

    idx = [argmin(abs.(heat.tspan .- t)) for t in times]
    Usel = integrate_full_order_at_indices(heat, u0, idx)

    return Usel, idx
end

"""
    state_to_grid(heat::Heat2DParams, u::AbstractVector)

Reshape a flattened state vector into an `Nx × Ny` grid.
"""
function state_to_grid(heat::Heat2DParams, u::AbstractVector)
    @assert length(u) == heat.d "State vector must have length heat.d"
    return reshape(u, heat.Nx, heat.Ny)
end


"""
    observation_grid_indices(N, m)

Choose `m` approximately equally spaced grid indices in each spatial
direction on an `N × N` grid.

Returns a vector of `(i,j)` index pairs.
"""
function observation_grid_indices(N::Int, m::Int)
    @assert 1 <= m <= N "Need 1 <= m <= N"
    pts = round.(Int, range(1, N, length=m))
    return vec([(i, j) for j in pts, i in pts])
end

"""
    observation_linear_indices(N, m)

Convert the 2D grid observation pattern produced by
`observation_grid_indices(N, m)` into linear indices for a flattened
`N × N` state.
"""
function observation_linear_indices(N::Int, m::Int)
    grid_pts = observation_grid_indices(N, m)
    return [i + (j - 1) * N for (i, j) in grid_pts]
end

"""
    observation_matrix_2d(N, m; T=Float64, sparse_output=false)

Build an observation matrix that samples an approximately equally spaced
`m × m` grid from a flattened `N × N` state.

Returns a matrix of size `(m^2, N^2)`.
"""
function observation_matrix_2d(
    N::Int,
    m::Int;
    T::Type{<:Real}=Float64,
    sparse_output::Bool=false,
)
    d = N^2
    idx = observation_linear_indices(N, m)
    d_out = length(idx)

    if sparse_output
        C = spzeros(T, d_out, d)
        for k in 1:d_out
            C[k, idx[k]] = one(T)
        end
    else
        C = zeros(T, d_out, d)
        for k in 1:d_out
            C[k, idx[k]] = one(T)
        end
    end

    return C
end

"""
    top_left_observation_indices(d, n)

Linear indices of the top-left `n × n` block of a flattened square state
of dimension `d = N^2`.
"""
function top_left_observation_indices(d::Int, n::Int)
    N = isqrt(d)
    @assert N^2 == d "State dimension d must be a perfect square"
    @assert 1 <= n <= N "Need 1 <= n <= N"

    return vec([i + (j - 1) * N for j in 1:n, i in 1:n])
end

"""
    top_left_observation_matrix(d, n; T=Float64, sparse_output=false)

Observation matrix selecting the top-left `n × n` block of a flattened
square state.
"""
function top_left_observation_matrix(
    d::Int,
    n::Int;
    T::Type{<:Real}=Float64,
    sparse_output::Bool=false,
)
    idx = top_left_observation_indices(d, n)
    d_out = length(idx)

    if sparse_output
        C = spzeros(T, d_out, d)
        for k in 1:d_out
            C[k, idx[k]] = one(T)
        end
    else
        C = zeros(T, d_out, d)
        for k in 1:d_out
            C[k, idx[k]] = one(T)
        end
    end

    return C
end

"""
    two_block_observation_indices(d, n, r)

Linear indices of the union of:
- the top-left `n × n` block
- the bottom-right `r × r` block

for a flattened square state of dimension `d = N^2`.
"""
function two_block_observation_indices(d::Int, n::Int, r::Int)
    N = isqrt(d)
    @assert N^2 == d "State dimension d must be a perfect square"
    @assert 1 <= n <= N "Need 1 <= n <= N"
    @assert 1 <= r <= N "Need 1 <= r <= N"

    idx_tl = vec([i + (j - 1) * N for j in 1:n, i in 1:n])
    idx_br = vec([i + (j - 1) * N for j in (N-r+1):N, i in (N-r+1):N])

    return unique(vcat(idx_tl, idx_br))
end

"""
    two_block_observation_matrix(d, n, r; T=Float64, sparse_output=false)

Observation matrix selecting the union of the top-left `n × n` block and
bottom-right `r × r` block of a flattened square state.
"""
function two_block_observation_matrix(
    d::Int,
    n::Int,
    r::Int;
    T::Type{<:Real}=Float64,
    sparse_output::Bool=false,
)
    idx = two_block_observation_indices(d, n, r)
    d_out = length(idx)

    if sparse_output
        C = spzeros(T, d_out, d)
        for k in 1:d_out
            C[k, idx[k]] = one(T)
        end
    else
        C = zeros(T, d_out, d)
        for k in 1:d_out
            C[k, idx[k]] = one(T)
        end
    end

    return C
end

"""
    observation_time_indices(heat, h)

Return observation indices in `heat.tspan` corresponding to measurements
every `h` units of time, starting at `h` and ending at `heat.T_stop`.

Returns `(obs_idx, obs_times)`.
"""
function observation_time_indices(heat::Heat2DParams{T}, h::T) where {T<:AbstractFloat}
    @assert h > zero(T) "Observation spacing h must be positive"

    skips = Int(round(h / heat.Δt))
    @assert skips >= 1 "Observation spacing too small relative to Δt"

    obs_idx = collect(1 + skips : skips : heat.time_dim)
    obs_times = heat.tspan[obs_idx]

    return obs_idx, obs_times
end

"""
    observe_trajectory(C, U, obs_idx)

Apply observation matrix `C` to the state trajectory `U` at the time
indices `obs_idx`.

Assumes `U` has size `(d, nt)` with columns equal to time snapshots.

Returns:
- `Y`    : matrix of size `(d_out, n_obs)`
- `yvec` : vectorized observations obtained by stacking columns of `Y`
"""
function observe_trajectory(
    C::AbstractMatrix,
    U::AbstractMatrix,
    obs_idx::AbstractVector{<:Integer},
)
    @assert all(1 .<= obs_idx .<= size(U, 2)) "Observation indices out of bounds"
    Y = C * U[:, obs_idx]
    yvec = vec(Y)
    return Y, yvec
end

"""
    observe_initial_condition(heat, C, u0, obs_idx)

Integrate the full-order model from initial condition `u0` and apply the
observation operator `C` at times indexed by `obs_idx`.

Returns:
- `U`    : full state trajectory
- `Y`    : observation matrix over selected times
- `yvec` : vectorized observation vector
"""
function observe_initial_condition(
    heat::Heat2DParams,
    C::AbstractMatrix,
    u0::AbstractVector,
    obs_idx::AbstractVector{<:Integer},
)
    U = integrate_full_order(heat, u0)
    Y, yvec = observe_trajectory(C, U, obs_idx)
    return U, Y, yvec
end

"""
    add_gaussian_noise(yclean; rel_noise_level=0.2, rng=Random.default_rng())

Add isotropic Gaussian noise to `yclean`, with standard deviation

    σ = rel_noise_level * maximum(abs.(yclean))

Returns `(y, Γ, ϵ, σ)`, where:
- `y` : noisy data
- `Γ` : noise covariance matrix
- `ϵ` : sampled noise vector
- `σ` : scalar standard deviation
"""
function add_gaussian_noise(
    yclean::AbstractVector{T};
    rel_noise_level::T=T(0.2),
    rng=Random.default_rng(),
) where {T<:AbstractFloat}

    σ = rel_noise_level * maximum(abs.(yclean))
    n = length(yclean)
    Γ = (σ^2) * I(n)
    ϵ = rand(rng, MvNormal(zeros(T, n), Symmetric(Matrix(Γ))))
    y = yclean + ϵ

    return y, Γ, ϵ, σ
end

"""
    build_synthetic_data(heat, C, u0, h; rel_noise_level=0.2, rng=Random.default_rng())

Generate synthetic observation data by:
1. integrating from initial condition `u0`
2. observing every `h` time units using `C`
3. adding isotropic Gaussian noise

Returns a named tuple containing trajectory, clean data, noisy data,
noise covariance, and observation timing information.
"""
function build_synthetic_data(
    heat::Heat2DParams{T},
    C::AbstractMatrix,
    u0::AbstractVector,
    h::T;
    rel_noise_level::T=T(0.2),
    rng=Random.default_rng(),
) where {T<:AbstractFloat}

    obs_idx, obs_times = observation_time_indices(heat, h)
    U, Y_clean, y_clean = observe_initial_condition(heat, C, u0, obs_idx)
    y, Γ, ϵ, σ = add_gaussian_noise(y_clean; rel_noise_level=rel_noise_level, rng=rng)

    return (
        U = U,
        C = C,
        obs_idx = obs_idx,
        obs_times = obs_times,
        Y_clean = Y_clean,
        y_clean = y_clean,
        y = y,
        Γ = Γ,
        ϵ = ϵ,
        σ = σ,
        n = length(y),
        d_out = size(C, 1),
        n_obs = length(obs_idx),
    )
end


"""
    build_explicit_forward_operator(heat, C, obs_idx)

Construct the explicit linear forward operator `H` mapping an initial
condition `v` to the stacked observation vector

    y = vec(C * U[:, obs_idx])

for the backward Euler full-order model.

Returns a dense matrix `H` of size `(n, d)`, where
- `d = heat.d`
- `n = size(C,1) * length(obs_idx)`
"""
function build_explicit_forward_operator(
    heat::Heat2DParams{T},
    C::AbstractMatrix,
    obs_idx::AbstractVector{<:Integer},
) where {T<:AbstractFloat}

    @assert all(1 .<= obs_idx .<= heat.time_dim) "Observation indices out of bounds"

    d = heat.d
    d_out = size(C, 1)
    n_obs = length(obs_idx)
    n = d_out * n_obs

    # Backward Euler one-step propagator:
    # u_{k+1} = (I - Δt A)^{-1} u_k
    F = I - heat.Δt * heat.A
    M = F \ I

    H = zeros(T, n, d)

    Mpow = Matrix{T}(I, d, d)
    p = 1
    obs_set = Set(obs_idx)

    for k in 1:heat.time_dim
        if k > 1
            Mpow = M * Mpow
        end

        if k in obs_set
            H[p:p+d_out-1, :] .= C * Mpow
            p += d_out
        end
    end

    return H
end

"""
    build_RLS_data(y, Γ, Γpr)

Build the regularized least-squares data used by EKRMLE:
- `ΓRLS = blockdiag(Γ, Γpr)`
- `yRLS = [y; zeros(d)]`

Returns `(yRLS, ΓRLS)`.
"""
function build_RLS_data(
    y::AbstractVector{T},
    Γ::AbstractMatrix,
    Γpr::AbstractMatrix,
) where {T<:AbstractFloat}

    n = length(y)
    d = size(Γpr, 1)

    @assert size(Γ, 1) == n && size(Γ, 2) == n "Γ must be n×n"
    @assert size(Γpr, 1) == d && size(Γpr, 2) == d "Γpr must be d×d"

    ΓRLS = zeros(T, n + d, n + d)
    ΓRLS[1:n, 1:n] .= Γ
    ΓRLS[n+1:end, n+1:end] .= Γpr

    yRLS = vcat(y, zeros(T, d))

    return yRLS, ΓRLS
end

"""
    build_explicit_HRLS_operator(H)

Given an explicit forward matrix `H`, return:
- `HRLS`: the dense matrix `[H; I]`
- `HRLS_s`: callable wrapper satisfying `HRLS_s(::Nothing, v) = HRLS * v`
"""
function build_explicit_HRLS_operator(H::AbstractMatrix{T}) where {T<:AbstractFloat}
    d = size(H, 2)
    HRLS = vcat(H, Matrix{T}(I, d, d))

    function HRLS_s(::Nothing, v::AbstractVector)
        return HRLS * v
    end

    return HRLS, HRLS_s
end

"""
    build_implicit_forward_operator(heat, C, obs_idx)

Return a callable forward map `H_of_v(v)` that:
1. integrates the full-order heat equation from initial condition `v`
2. extracts snapshots at `obs_idx`
3. applies `C`
4. stacks observations into a vector

This avoids forming the explicit matrix `H`.
"""
function build_implicit_forward_operator(
    heat::Heat2DParams,
    C::AbstractMatrix,
    obs_idx::AbstractVector{<:Integer},
)
    @assert all(1 .<= obs_idx .<= heat.time_dim) "Observation indices out of bounds"

    function H_of_v(v::AbstractVector)
        U = integrate_full_order(heat, v)
        _, yvec = observe_trajectory(C, U, obs_idx)
        return yvec
    end

    return H_of_v
end

"""
    build_implicit_HRLS_operator(heat, C, obs_idx)

Return:
- `H_of_v`: implicit forward map from initial condition to stacked observations
- `HRLS_s`: callable regularized forward map with output `[H(v); v]`
"""
function build_implicit_HRLS_operator(
    heat::Heat2DParams,
    C::AbstractMatrix,
    obs_idx::AbstractVector{<:Integer},
)
    H_of_v = build_implicit_forward_operator(heat, C, obs_idx)

    function HRLS_s(::Nothing, v::AbstractVector)
        return vcat(H_of_v(v), v)
    end

    return H_of_v, HRLS_s
end


mutable struct ReducedHeatOperators{
    T<:AbstractFloat,
    TI<:Int,
    M1<:AbstractMatrix{T},
    M2<:AbstractMatrix{T},
    M3<:AbstractMatrix{T},
    M4<:AbstractMatrix{T},
    M5<:AbstractMatrix{T},
    M6<:AbstractMatrix{T}
}
    r::TI
    sigobs::Vector{T}

    # Factors used in reduction
    R::M1
    L::M2

    # Truncation SVD
    Ur::M3
    Vr::M4
    Sr::Vector{T}

    # Trial / test bases
    Wr::M5
    Wr_tild_tr::M6

    # Reduced operators
    Ahat::Matrix{T}
    Bhat::Matrix{T}
    Chat::Matrix{T}
end

"""
    build_reduction_factors(heat, C, Γ, r)

Construct the balancing-style reduction ingredients using

- `R` from the prior covariance `Γpr = R'R`
- `L` from the Lyapunov factor associated with `(A', F')`
  where `F = C ./ sigobs`

and truncate to rank `r`.

Returns a named tuple containing the raw factors and SVD pieces.
"""
function build_reduction_factors(
    heat::Heat2DParams{T},
    C::AbstractMatrix,
    Γ::AbstractMatrix,
    r::Int,
) where {T<:AbstractFloat}

    d_out = size(C, 1)
    @assert size(Γ, 1) >= d_out && size(Γ, 2) >= d_out "Γ must contain at least one observation block"
    @assert 1 <= r <= heat.d "Reduction rank r must satisfy 1 <= r <= heat.d"

    # Use first observation block to normalize rows of C
    sigobs = sqrt.(diag(Γ[1:d_out, 1:d_out]))
    F = C ./ sigobs

    # Prior factor: Γpr = R'R
    R = cholesky(Symmetric(heat.Γpr)).U

    # Observability-like factor from Lyapunov equation
    # Q solves A'Q + QA + F'F = 0
    # L is a factor with Q = L'L
    L = plyapc(Matrix(heat.A)', Matrix(F)')'

    # SVD of L'R
    U, S, V = svd(L' * R)

    @assert r <= length(S) "Reduction rank exceeds available singular values"

    Ur = U[:, 1:r]
    Vr = V[:, 1:r]
    Sr = S[1:r]

    return (
        sigobs = sigobs,
        R = R,
        L = L,
        Ur = Ur,
        Vr = Vr,
        Sr = Sr,
    )
end

"""
    build_reduced_operators(heat, C, Γ; r)

Build the reduced-order operators using the balancing-style truncation.

Returns a `ReducedHeatOperators` object containing:
- `Wr`
- `Wr_tild_tr`
- `Ahat`, `Bhat`, `Chat`
and the intermediate factors.
"""
function build_reduced_operators(
    heat::Heat2DParams{T},
    C::AbstractMatrix,
    Γ::AbstractMatrix;
    r::Int,
) where {T<:AbstractFloat}

    fac = build_reduction_factors(heat, C, Γ, r)

    Sigr = Diagonal(inv.(sqrt.(fac.Sr)))

    Wr_tild_tr = Sigr * fac.Ur' * fac.L'
    Wr = fac.R * fac.Vr * Sigr

    Ahat = Matrix(Wr_tild_tr * heat.A * Wr)
    Bhat = Matrix(Wr_tild_tr * heat.B)
    Chat = Matrix(C * Wr)

    return ReducedHeatOperators(
        r,
        fac.sigobs,
        fac.R,
        fac.L,
        fac.Ur,
        fac.Vr,
        fac.Sr,
        Wr,
        Wr_tild_tr,
        Ahat,
        Bhat,
        Chat,
    )
end

"""
    build_reduced_solver(red, Δt)

Construct the fast dense reduced solver from `Ahat`.
"""
function build_reduced_solver(
    red::ReducedHeatOperators,
    Δt::T,
) where {T<:AbstractFloat}
    return FastDenseSolver(red.Ahat, Δt)
end

"""
    build_reduced_explicit_forward_operator(heat, red, obs_idx)

Construct the explicit reduced forward operator

    Hhat * v ≈ vec(Chat * Uhat[:, obs_idx])

where the reduced initial condition is `Wr_tild_tr * v`.

Returns a dense matrix `Hhat` of size `(n, d)`.
"""
function build_reduced_explicit_forward_operator(
    heat::Heat2DParams{T},
    red::ReducedHeatOperators{T},
    obs_idx::AbstractVector{<:Integer},
) where {T<:AbstractFloat}

    @assert all(1 .<= obs_idx .<= heat.time_dim) "Observation indices out of bounds"

    d = heat.d
    d_out = size(red.Chat, 1)
    n_obs = length(obs_idx)
    n = d_out * n_obs

    r = red.r

    Fhat = I - heat.Δt * red.Ahat
    Mhat = Fhat \ I

    Hhat = zeros(T, n, d)

    Mpowhat = Matrix{T}(I, r, r)
    p = 1
    obs_set = Set(obs_idx)

    for k in 1:heat.time_dim
        if k > 1
            Mpowhat = Mhat * Mpowhat
        end

        if k in obs_set
            Hhat[p:p+d_out-1, :] .= red.Chat * Mpowhat * red.Wr_tild_tr
            p += d_out
        end
    end

    return Hhat
end

"""
    build_reduced_implicit_forward_operator(heat, red, obs_idx)

Return a callable reduced forward map `Hhat_of_v(v)` that:
1. projects `v` to reduced coordinates via `Wr_tild_tr * v`
2. integrates the reduced system
3. applies `Chat` at observation times
4. stacks the observations into a vector
"""
function build_reduced_implicit_forward_operator(
    heat::Heat2DParams,
    red::ReducedHeatOperators,
    obs_idx::AbstractVector{<:Integer},
)
    @assert all(1 .<= obs_idx .<= heat.time_dim) "Observation indices out of bounds"

    red_solver = build_reduced_solver(red, heat.Δt)

    function Hhat_of_v(v::AbstractVector)
        vhat0 = red.Wr_tild_tr * v
        Uhat = integrate_model_fast(red_solver, heat.tspan, vhat0, red.Bhat, heat.Ubc)

        y = Vector{Float64}(undef, length(obs_idx) * size(red.Chat, 1))

        p = 1
        for k in obs_idx
            obs = red.Chat * Uhat[:, k]
            m = length(obs)
            y[p:p+m-1] = obs
            p += m
        end

        return y
    end

    return Hhat_of_v, red_solver
end

"""
    build_reduced_HRLS_operator(heat, red, obs_idx)

Return:
- `Hhat_of_v`
- `HRLShat_s`

where

    HRLShat_s(::Nothing, v) = [Hhat_of_v(v); v]

so that EKRMLE still evolves in the full space.
"""
function build_reduced_HRLS_operator(
    heat::Heat2DParams,
    red::ReducedHeatOperators,
    obs_idx::AbstractVector{<:Integer},
)
    Hhat_of_v, red_solver = build_reduced_implicit_forward_operator(heat, red, obs_idx)

    function HRLShat_s(::Nothing, v::AbstractVector)
        return vcat(Hhat_of_v(v), v)
    end

    return Hhat_of_v, HRLShat_s, red_solver
end


"""
    weighted_norm(x, C)

Compute the weighted norm

    ‖x‖_{C^{-1}} = sqrt(x' * C^{-1} * x)

Assumes `C` is square and invertible (typically SPD).
"""
function weighted_norm(x::AbstractVector, C::AbstractMatrix)
    @assert size(C, 1) == size(C, 2) "C must be square"
    @assert length(x) == size(C, 1) "Dimension mismatch between x and C"

    return sqrt(dot(x, C \ x))
end

"""
    weighted_relative_error(A, B, C)

Compute the relative error

    ‖A - B‖_{C^{-1}} / ‖A‖_{C^{-1}}

where
    ‖x‖_{C^{-1}} = sqrt(x' * C^{-1} * x).

Assumes `C` is square and invertible.
"""
function weighted_relative_error(
    A::AbstractVector,
    B::AbstractVector,
    C::AbstractMatrix
)
    @assert length(A) == length(B) "A and B must have the same length"
    @assert size(C, 1) == size(C, 2) == length(A) "Dimension mismatch"

    denom = weighted_norm(A, C)
    @assert denom > 0 "Weighted norm of A is zero, relative error undefined"

    return weighted_norm(A - B, C) / denom
end