using EKRMLE
using LinearAlgebra
using CairoMakie
using PolynomialModelReductionDataset: Heat2DModel, integrate_model_fast,
    build_fast_be_solver, FastDenseSolver
using UniqueKronecker: invec
using Random
using Distributions
using MatrixEquations
using ColorSchemes
using SparseArrays

CairoMakie.activate!()

##
# ------------------------------------------------------------
# Helper: exact Gaussian posterior
# ------------------------------------------------------------

function posterior_stats(
    H::AbstractMatrix,
    y::AbstractVector,
    Γ::AbstractMatrix,
    Γpr::AbstractMatrix,
)
    Γpos = (H' * (Γ \ H) + (Γpr \ I)) \ I
    μpos = Γpos * H' * (Γ \ y)

    return μpos, Γpos
end

##
# ------------------------------------------------------------
# 1. Problem setup
# ------------------------------------------------------------

rng = MersenneTwister(1234)

Nx = 64
Ny = 64

heat = build_heat2d_params(
    Nx = Nx,
    Ny = Ny,
    Δt = 1e-2,
    T_stop = 2.0,
    diffusion_coeffs = 0.5,
)

# Observation operator
C = two_block_observation_matrix(heat.d, 50, 50)

# Draw true initial condition
u_true = sample_prior_ic(heat; rng=rng)

# Synthetic observations
data = build_synthetic_data(
    heat,
    C,
    u_true,
    0.1;
    rel_noise_level = 0.02,
    rng = rng,
)

# RLS quantities used by EKRMLE
yRLS, ΓRLS = build_RLS_data(
    data.y,
    data.Γ,
    heat.Γpr,
)

##
# ------------------------------------------------------------
# 2. Full-order Bayesian posterior
# ------------------------------------------------------------

H_full = build_explicit_forward_operator(
    heat,
    C,
    data.obs_idx,
)

μpos_full, Γpos_full = posterior_stats(
    H_full,
    data.y,
    data.Γ,
    heat.Γpr,
)

##
# ------------------------------------------------------------
# 3. reduced Bayesian posterior
# ------------------------------------------------------------

r = 100

red = build_reduced_operators(
    heat,
    C,
    data.Γ;
    r = r,
)

Hhat = build_reduced_explicit_forward_operator(
    heat,
    red,
    data.obs_idx,
)

μpos_red, Γpos_red = posterior_stats(
    Hhat,
    data.y,
    data.Γ,
    heat.Γpr,
)


##
# ------------------------------------------------------------
# 5. Convert parameter vectors to 2D fields
# ------------------------------------------------------------

u_true_2d    = reshape(u_true, Nx, Ny)
μpos_full_2d = reshape(μpos_full, Nx, Ny)
μpos_red_2d  = reshape(μpos_red, Nx, Ny)

# Common color range for a fair visual comparison
allvals = vcat(
    vec(u_true_2d),
    vec(μpos_full_2d),
    vec(μpos_red_2d),
)

crange = extrema(allvals)

##
# ------------------------------------------------------------
# 6. Plot
# ------------------------------------------------------------

fig = Figure(size = (1800, 500))

ax11 = Axis(
    fig[1, 1],
    title = L"\text{truth}",
    titlesize = 50,
    aspect = DataAspect(),
)

ax12 = Axis(
    fig[1, 2],
    title = L"\text{posterior mean}",
    titlesize = 50,
    aspect = DataAspect(),
)

ax13 = Axis(
    fig[1, 3],
    title = L"\text{BT posterior mean}",
    titlesize = 50,
    aspect = DataAspect(),
)

h11 = heatmap!(
    ax11,
    u_true_2d;
    colorrange = crange,
    colormap = :magma,
)

heatmap!(
    ax12,
    μpos_full_2d;
    colorrange = crange,
    colormap = :magma,
)

heatmap!(
    ax13,
    μpos_red_2d;
    colorrange = crange,
    colormap = :magma,
)

# Hide redundant tick labels
hidedecorations!(ax11)
hidedecorations!(ax12)
hidedecorations!(ax13)
# Shared colorbar
Colorbar(
    fig[:, 4],
    h11,
)

fig


save("plots/Heat_2D_sbs.pdf",fig)

##

