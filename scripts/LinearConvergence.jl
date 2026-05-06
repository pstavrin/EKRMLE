using EnsembleKalmanRMLE
using LinearAlgebra
using CairoMakie
using Random
using Distributions
using MatrixEquations
using ColorSchemes
CairoMakie.activate!()


@inline colmean(V::AbstractMatrix) = vec(mean(V, dims=2))

function _samplecov(V::AbstractMatrix)
    # computes sample covariance (column-wise)
    J = size(V, 2)
    μ = colmean(V)
    X = V .- μ
    return (X * X') / (J-1)
end

## Setup
n = 500
d = 1000
J = 100000
prob = randomLinearProblemObj(n, d, J)

## Errors
struct MisfitPair{T<:AbstractVector}
    mean::T
    cov::T
end

struct ProjectedMisfits{T<:AbstractVector}
    calP::MisfitPair{T}  # 𝒫 metric in observation space
    P::MisfitPair{T}     # ℙ metric in parameter/state space
end


function normalized_proj_error(A, limit, P; atol=eps(real(eltype(A))))
    PA_diff = P * (A - limit)
    Plimit = P * limit

    err = zero(real(eltype(A)))
    J = size(A, 2)

    @views for j in 1:J
        denom = norm(Plimit[:, j])
        numer = norm(PA_diff[:, j])

        err += denom > atol ? numer / denom : numer
    end

    return err / J
end


function getLimitMisfits_P_calP(ỹ, V, iters, H, Γ, Hₚ, H⁺, 𝒫, ℙ)
    # Allocate output arrays
    𝒫norms    = zeros(iters)
    𝒫covnorms = zeros(iters)

    ℙnorms    = zeros(iters)
    ℙcovnorms = zeros(iters)

    # Initial ensemble
    v₀ = V[1]
    h₀ = H * v₀

    # Limiting ensembles
    v_inf = ℙ * H⁺ * ỹ
    h_inf = 𝒫 * ỹ

    # If your original limiting formulas require the nullspace terms,
    # use these instead:
    #
    # v_inf = ℙ * H⁺ * ỹ + (I - ℙ) * v₀
    # h_inf = 𝒫 * ỹ + (I - 𝒫) * h₀

    # Limiting covariances
    C_inf   = _samplecov(v_inf)
    HGH_inf = _samplecov(h_inf)

    # Projected operators
    𝒫H = 𝒫 * H

    # Projected limiting covariances
    𝒫HGH_inf = 𝒫 * HGH_inf * 𝒫'
    ℙC_inf   = ℙ * C_inf * ℙ'

    # Normalization constants
    𝒫HGH_norm = opnorm(𝒫HGH_inf)
    ℙC_norm   = opnorm(ℙC_inf)

    # Avoid division by zero if the limiting covariance vanishes
    𝒫HGH_norm = 𝒫HGH_norm > 0 ? 𝒫HGH_norm : one(𝒫HGH_norm)
    ℙC_norm   = ℙC_norm > 0 ? ℙC_norm : one(ℙC_norm)

    for i in 1:iters
        vᵢ = V[i]
        hᵢ = H * vᵢ
        Cᵢ = _samplecov(vᵢ)

        # Mean / particle misfit
        𝒫norms[i] = normalized_proj_error(hᵢ, h_inf, 𝒫)
        ℙnorms[i] = normalized_proj_error(vᵢ, v_inf, ℙ)

        # Covariance misfit
        𝒫covnorms[i] = opnorm(𝒫H * Cᵢ * 𝒫H' - 𝒫HGH_inf) / 𝒫HGH_norm
        ℙcovnorms[i] = opnorm(ℙ * Cᵢ * ℙ' - ℙC_inf) / ℙC_norm
    end

    return ProjectedMisfits(
        MisfitPair(𝒫norms, 𝒫covnorms),
        MisfitPair(ℙnorms, ℙcovnorms),
    )
end

## EKRMLE
V0 = prob.V0
steps = 100
obj = EKRMLEObj(V0, prob.y, prob.Γ)
H_s(prob::randomLinearProblemObj, v::AbstractVector) = prob.H * v # forward map
#EKRMLE_run!(obj, prob, H_s, steps); # run algorithm

## Spectral projectors
projectors = spectralproj(prob, _samplecov(V0))
P = real.(projectors.P);
calP = real.(projectors.calP)

##

M = getLimitMisfits_P_calP(obj.Yrand,obj.V,steps,prob.H,obj.Γ, prob.pHess, prob.H⁺, calP, P)

## Experimental loop
function run_ensemble_size_study(prob, ensemble_sizes; steps=100)
    # Storage
    results = Dict{Int,Any}()

    # Forward map
    H_s(prob::randomLinearProblemObj, v::AbstractVector) = prob.H * v

    for J in ensemble_sizes
        println("Running EKRMLE with J = $J")

        # ------------------------------------------------------------
        # 1. Restrict initial ensemble
        # ------------------------------------------------------------
        V0_J = prob.V0[:, 1:J]

        # ------------------------------------------------------------
        # 2. Run EKRMLE
        # ------------------------------------------------------------
        obj_J = EKRMLEObj(V0_J, prob.y, prob.Γ)
        EKRMLE_run!(obj_J, prob, H_s, steps)

        # ------------------------------------------------------------
        # 3. Construct spectral projectors from initial ensemble
        # ------------------------------------------------------------
        projectors_J = spectralproj(prob, _samplecov(V0_J))

        P_J    = real.(projectors_J.P)
        calP_J = real.(projectors_J.calP)

        # ------------------------------------------------------------
        # 4. Compute projected limit misfits
        # ------------------------------------------------------------
        M_J = getLimitMisfits_P_calP(
            obj_J.Yrand,
            obj_J.V,
            steps,
            prob.H,
            obj_J.Γ,
            prob.pHess,
            prob.H⁺,
            calP_J,
            P_J,
        )

        # ------------------------------------------------------------
        # 5. Store everything relevant
        # ------------------------------------------------------------
        results[J] = (
            obj = obj_J,
            projectors = projectors_J,
            P = P_J,
            calP = calP_J,
            misfits = M_J,

            calP_mean_errors = M_J.calP.mean,
            calP_cov_errors  = M_J.calP.cov,

            P_mean_errors = M_J.P.mean,
            P_cov_errors  = M_J.P.cov,
        )
    end

    return results
end

## run experiment
ensemble_sizes = [10, 100, 1000, 5000, 10000, 50000]

results = run_ensemble_size_study(prob, ensemble_sizes; steps=100)

## plots

colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=length(ensemble_sizes)+2)]
colors = colors[end:-1:1]
X = range(0,steps,length = steps)

fig = Figure(size=(1000,800))

ax1 = Axis(fig[1,1];
    xlabel=L"Iterations $i$",
    title=L"\text{Rel. }\text{\textbf{h}}_i^{(1:J)} \text{ error }(\mathcal{P} \text{ space})",
    #title=L"𝓟\text{ space convergence}",
    yscale=log10,
    xlabelsize = 0,
    ylabelsize = 30,
    titlesize = 30,
    xticklabelsize = 0,
    yticklabelsize = 28,
)


# Plot lines
lines!(ax1, X, vec(results[10].calP_mean_errors), label=L"J=10", color=colors[2], linewidth=7)
lines!(ax1, X, vec(results[100].calP_mean_errors), label=L"J=100", color=colors[3], linewidth=7)
lines!(ax1, X, vec(results[1000].calP_mean_errors), label=L"J=1000", color=colors[4], linewidth=7)
lines!(ax1, X, vec(results[5000].calP_mean_errors), label=L"J=5000", color=colors[5], linewidth=7)
lines!(ax1, X, vec(results[10000].calP_mean_errors), label=L"J=10000", color=colors[6], linewidth=7)
lines!(ax1, X, vec(results[50000].calP_mean_errors), label=L"J=50000", color=colors[7], linewidth=7)
#axislegend(ax1, position=:lb, labelsize=28, framevisible=false)

ax2 = Axis(fig[1,2];
    xlabel=L"Iterations $i$",
    title=L"\text{Rel. }\mathrm{Cov}[\text{\textbf{h}}_i^{(1:J)}] \text{ error }(\mathcal{P} \text{ space})",
    #title=L"𝓟\text{ space convergence}",
    yscale=log10,
    xlabelsize = 0,
    ylabelsize = 0,
    titlesize = 30,
    xticklabelsize = 0,
    yticklabelsize = 0,
)


# Plot lines
lines!(ax2, X, vec(results[10].calP_cov_errors), label=L"J=10", color=colors[2], linewidth=7)
lines!(ax2, X, vec(results[100].calP_cov_errors), label=L"J=100", color=colors[3], linewidth=7)
lines!(ax2, X, vec(results[1000].calP_cov_errors), label=L"J=1000", color=colors[4], linewidth=7)
lines!(ax2, X, vec(results[5000].calP_cov_errors), label=L"J=5000", color=colors[5], linewidth=7)
lines!(ax2, X, vec(results[10000].calP_cov_errors), label=L"J=10000", color=colors[6], linewidth=7)
lines!(ax2, X, vec(results[50000].calP_cov_errors), label=L"J=50000", color=colors[7], linewidth=7)
axislegend(ax2, position=:lb, labelsize=25, framevisible=false)


ax3 = Axis(fig[2,1];
    xlabel=L"Iterations $i$",
    title=L"\text{Rel. }\text{\textbf{v}}_i^{(1:J)} \text{ error }(\textbf{P} \text{ space})",
    #title=L"𝓟\text{ space convergence}",
    yscale=log10,
    xlabelsize = 30,
    ylabelsize = 30,
    titlesize = 30,
    xticklabelsize = 30,
    yticklabelsize = 28,
)


# Plot lines
lines!(ax3, X, vec(results[10].P_mean_errors), label=L"J=10", color=colors[2], linewidth=7)
lines!(ax3, X, vec(results[100].P_mean_errors), label=L"J=100", color=colors[3], linewidth=7)
lines!(ax3, X, vec(results[1000].P_mean_errors), label=L"J=1000", color=colors[4], linewidth=7)
lines!(ax3, X, vec(results[5000].P_mean_errors), label=L"J=5000", color=colors[5], linewidth=7)
lines!(ax3, X, vec(results[10000].P_mean_errors), label=L"J=10000", color=colors[6], linewidth=7)
lines!(ax3, X, vec(results[50000].P_mean_errors), label=L"J=50000", color=colors[7], linewidth=7)
#axislegend(ax3, position=:lb, labelsize=28, framevisible=false)

ax4 = Axis(fig[2,2];
    xlabel=L"Iterations $i$",
    title=L"\text{Rel. }\mathrm{Cov}[\text{\textbf{v}}_i^{(1:J)}] \text{ error }(\textbf{P} \text{ space})",
    #title=L"𝓟\text{ space convergence}",
    yscale=log10,
    xlabelsize = 30,
    ylabelsize = 0,
    titlesize = 30,
    xticklabelsize = 30,
    yticklabelsize = 0,
)


# Plot lines
lines!(ax4, X, vec(results[10].P_cov_errors), label=L"J=10", color=colors[2], linewidth=7)
lines!(ax4, X, vec(results[100].P_cov_errors), label=L"J=100", color=colors[3], linewidth=7)
lines!(ax4, X, vec(results[1000].P_cov_errors), label=L"J=1000", color=colors[4], linewidth=7)
lines!(ax4, X, vec(results[5000].P_cov_errors), label=L"J=5000", color=colors[5], linewidth=7)
lines!(ax4, X, vec(results[10000].P_cov_errors), label=L"J=10000", color=colors[6], linewidth=7)
lines!(ax4, X, vec(results[50000].P_cov_errors), label=L"J=50000", color=colors[7], linewidth=7)
#axislegend(ax4, position=:lb, labelsize=28, framevisible=false)

linkyaxes!(ax1, ax2)
linkyaxes!(ax3, ax4)
display(fig)
#save("plots/convergence_moresizes.pdf", fig)

