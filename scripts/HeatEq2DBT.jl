using EnsembleKalmanRMLE
using LinearAlgebra
using CairoMakie
using PolynomialModelReductionDataset: Heat2DModel, integrate_model_fast, build_fast_be_solver, FastDenseSolver
using UniqueKronecker: invec
using Random
using Distributions
using MatrixEquations
using ColorSchemes
using SparseArrays
using ProgressMeter
using JLD2
using Dates
CairoMakie.activate!()

## Helpers 

"""
    sample_cov(V)

Sample covariance of an ensemble matrix `V`, where columns are ensemble members.
Returns a dense covariance matrix of size d×d.
"""
function sample_cov(V::AbstractMatrix)
    J = size(V, 2)
    @assert J >= 2 "Need at least 2 ensemble members to compute covariance"

    μ = vec(mean(V, dims=2))
    X = V .- μ
    return (X * X') / (J - 1)
end

"""
    posterior_stats(H, y, Γ, Γpr)

Compute Gaussian posterior mean and covariance for the linear inverse problem

    y = H v + η,   η ~ N(0, Γ),   v ~ N(0, Γpr)

Returns `(μpos, Γpos)`.
"""
function posterior_stats(
    H::AbstractMatrix,
    y::AbstractVector,
    Γ::AbstractMatrix,
    Γpr::AbstractMatrix,
)
    Γpos = (H' * (Γ \ H) + (Γpr \ I))\I
    μpos = Γpos * H' * (Γ \ y)
    return μpos, Γpos
end

"""
    ensemble_stats_from_obj(ekobj)

Extract ensemble mean and covariance from the final EKRMLE ensemble.
Assumes `ekobj.V[end]` is a matrix with columns equal to ensemble members.
"""
function ensemble_stats_from_obj(ekobj)
    Vfinal = ekobj.V[end]
    μ = vec(mean(Vfinal, dims=2))
    Γ = sample_cov(Vfinal)
    return μ, Γ
end

"""
    save_experiment_results(savepath, results)

Save results dictionary / named tuple to a JLD2 file.
"""
function save_experiment_results(savepath::AbstractString, results)
    @save savepath results
    return savepath
end

## Experimental loop
"""
    run_model_reduction_experiment(;
        Js,
        Rs,
        steps=25,
        rng=Random.default_rng(),
        savepath="heat2d_model_reduction_results.jld2",
        heat_kwargs...,
        obs_builder=nothing,
        obs_h=0.1,
        rel_noise_level=0.2
    )

Run the model reduction experiment for the 2D heat equation.

Workflow:
1. Set up one heat problem
2. Generate one truth / synthetic dataset
3. Compute full posterior statistics
4. Precompute reduced models and reduced posterior statistics for each rank
5. For each ensemble size J:
    - run full-order EKRMLE
    - compute full-order errors vs full posterior
    - for each rank r:
        - run reduced-order EKRMLE
        - compute EKRMLE errors vs full posterior
        - record BT errors vs full posterior
6. Save all results

Arguments:
- `Js`: vector of ensemble sizes
- `Rs`: vector of reduced ranks
- `steps`: EKRMLE iterations (default 25)
- `obs_builder`: function `obs_builder(heat) -> C`; if `nothing`, uses `two_block_observation_matrix(heat.d, 5, 5)`
- `obs_h`: observation spacing in time
- `rel_noise_level`: relative noise level for synthetic data
- `heat_kwargs...`: forwarded to `build_heat2d_params`

Returns a named tuple `results`.
"""
function run_model_reduction_experiment(;
    Js::AbstractVector{<:Integer},
    Rs::AbstractVector{<:Integer},
    steps::Int=25,
    rng=Random.default_rng(),
    savepath::AbstractString="heat2d_model_reduction_results.jld2",
    obs_builder=nothing,
    obs_h::Float64=0.1,
    rel_noise_level::Float64=0.2,
    heat_kwargs...,
)
    @assert all(Js .>= 2) "All ensemble sizes must be at least 2"
    @assert all(Rs .>= 1) "All reduced ranks must be positive"

    Js = collect(Int.(Js))
    Rs = collect(Int.(Rs))

    nJ = length(Js)
    nR = length(Rs)
    Jmax = maximum(Js)

    # --------------------------------------------------------
    # 1. Problem setup
    # --------------------------------------------------------
    heat = build_heat2d_params(; heat_kwargs...)

    C = isnothing(obs_builder) ? two_block_observation_matrix(heat.d, 5, 5) : obs_builder(heat)

    u_true = sample_prior_ic(heat; rng=rng)

    data = build_synthetic_data(
        heat,
        C,
        u_true,
        obs_h;
        rel_noise_level=rel_noise_level,
        rng=rng,
    )

    yRLS, ΓRLS = build_RLS_data(data.y, data.Γ, heat.Γpr)

    # --------------------------------------------------------
    # 2. Full posterior statistics
    # --------------------------------------------------------
    H_full = build_explicit_forward_operator(heat, C, data.obs_idx)
    μpos_full, Γpos_full = posterior_stats(H_full, data.y, data.Γ, heat.Γpr)

    # Implicit full-order HRLS operator
    ~, HRLS_s_imp = build_implicit_HRLS_operator(heat, C, data.obs_idx)

    # --------------------------------------------------------
    # 3. Precompute reduced models + reduced posterior stats
    # --------------------------------------------------------
    reduced_models = Vector{Any}(undef, nR)
    Hhat_explicit = Vector{Any}(undef, nR)
    μpos_red = Vector{Vector{Float64}}(undef, nR)
    Γpos_red = Vector{Matrix{Float64}}(undef, nR)

    bt_mean_err = zeros(Float64, nR)
    bt_cov_err  = zeros(Float64, nR)

    @showprogress desc="Precomputing reduced models" for ir in eachindex(Rs)
        r = Rs[ir]

        red = build_reduced_operators(heat, C, data.Γ; r=r)
        Hhat = build_reduced_explicit_forward_operator(heat, red, data.obs_idx)
        μhat, Γhat = posterior_stats(Hhat, data.y, data.Γ, heat.Γpr)

        reduced_models[ir] = red
        Hhat_explicit[ir] = Hhat
        μpos_red[ir] = μhat
        Γpos_red[ir] = Γhat

        bt_mean_err[ir] = weighted_relative_error(μpos_full, μhat, Γpos_full)
        bt_cov_err[ir]  = opnorm(Γpos_full - Γhat) / opnorm(Γpos_full)
    end

    # --------------------------------------------------------
    # 4. Storage arrays
    # --------------------------------------------------------
    full_mean_err = zeros(Float64, nJ)
    full_cov_err  = zeros(Float64, nJ)
    full_time     = zeros(Float64, nJ)

    red_mean_err = zeros(Float64, nJ, nR)
    red_cov_err  = zeros(Float64, nJ, nR)
    red_time     = zeros(Float64, nJ, nR)

    # Optional: store BT errors duplicated across J for convenience
    bt_mean_err_by_J = repeat(reshape(bt_mean_err, 1, nR), nJ, 1)
    bt_cov_err_by_J  = repeat(reshape(bt_cov_err, 1, nR), nJ, 1)

    # --------------------------------------------------------
    # 5. Common initial ensemble pool
    # --------------------------------------------------------
    V0_pool = rand(rng, MvNormal(zeros(heat.d), Symmetric(heat.Γpr)), Jmax)

    # --------------------------------------------------------
    # 6. Main experiment loop
    # --------------------------------------------------------
    outer = Progress(nJ; desc="Ensemble sizes")

    for (ij, J) in pairs(Js)
        V0 = V0_pool[:, 1:J]

        # ----------------------------
        # Full-order EKRMLE
        # ----------------------------
        ek_full = EKRMLEObj(copy(V0), yRLS, ΓRLS)

        t_full = @elapsed EKRMLE_run!(ek_full, nothing, HRLS_s_imp, steps)
        μek_full, Γek_full = ensemble_stats_from_obj(ek_full)

        full_time[ij] = t_full
        full_mean_err[ij] = weighted_relative_error(μpos_full, μek_full, Γpos_full)
        full_cov_err[ij]  = opnorm(Γpos_full - Γek_full) / opnorm(Γpos_full)

        # ----------------------------
        # Reduced-order EKRMLE
        # ----------------------------
        inner = Progress(nR; desc="Ranks for J=$J", enabled=true)

        for ir in eachindex(Rs)
            red = reduced_models[ir]

            _, HRLShat_s_imp, _ = build_reduced_HRLS_operator(heat, red, data.obs_idx)

            ek_red = EKRMLEObj(copy(V0), yRLS, ΓRLS)

            t_red = @elapsed EKRMLE_run!(ek_red, nothing, HRLShat_s_imp, steps)
            μek_red, Γek_red = ensemble_stats_from_obj(ek_red)

            red_time[ij, ir] = t_red
            red_mean_err[ij, ir] = weighted_relative_error(μpos_full, μek_red, Γpos_full)
            red_cov_err[ij, ir]  = opnorm(Γpos_full - Γek_red) / opnorm(Γpos_full)

            next!(inner)
        end

        # Save checkpoint after each J
        results = (
            timestamp = string(now()),
            steps = steps,
            Js = Js,
            Rs = Rs,

            heat = heat,
            C = C,
            data = data,

            u_true = u_true,

            H_full = H_full,
            μpos_full = μpos_full,
            Γpos_full = Γpos_full,

            reduced_models = reduced_models,
            Hhat_explicit = Hhat_explicit,
            μpos_red = μpos_red,
            Γpos_red = Γpos_red,

            bt_mean_err = bt_mean_err,
            bt_cov_err = bt_cov_err,
            bt_mean_err_by_J = bt_mean_err_by_J,
            bt_cov_err_by_J = bt_cov_err_by_J,

            full_mean_err = full_mean_err,
            full_cov_err = full_cov_err,
            full_time = full_time,

            red_mean_err = red_mean_err,
            red_cov_err = red_cov_err,
            red_time = red_time,
        )

        save_experiment_results(savepath, results)
        next!(outer)
    end

    # Final save
    results = (
        timestamp = string(now()),
        steps = steps,
        Js = Js,
        Rs = Rs,

        heat = heat,
        C = C,
        data = data,

        u_true = u_true,

        H_full = H_full,
        μpos_full = μpos_full,
        Γpos_full = Γpos_full,

        reduced_models = reduced_models,
        Hhat_explicit = Hhat_explicit,
        μpos_red = μpos_red,
        Γpos_red = Γpos_red,

        bt_mean_err = bt_mean_err,
        bt_cov_err = bt_cov_err,
        bt_mean_err_by_J = bt_mean_err_by_J,
        bt_cov_err_by_J = bt_cov_err_by_J,

        full_mean_err = full_mean_err,
        full_cov_err = full_cov_err,
        full_time = full_time,

        red_mean_err = red_mean_err,
        red_cov_err = red_cov_err,
        red_time = red_time,
    )

    save_experiment_results(savepath, results)
    return results
end

## Run experiment
#=
Js = [1_000, 5_000, 10_000, 50_000]
Rs = [10, 20, 30, 50, 100]

results = run_model_reduction_experiment(
    Js=Js,
    Rs=Rs,
    steps=25,
    savepath="data/results_heat2d_bt_ekrmle.jld2",
    obs_h=0.1,
    rel_noise_level=0.2,
    Nx=32,
    Ny=32,
    Δt=1e-2,
    T_stop=2.0,
    diffusion_coeffs=0.5,
)
=#
## Run MC experiments
"""
    run_model_reduction_experiment_mc(;
        MC_runs,
        base_seed=1234,
        savepath="heat2d_model_reduction_results_mc.jld2",
        kwargs...
    )

Run `MC_runs` independent realizations of `run_model_reduction_experiment`
and aggregate errors/timings.

Returns a named tuple containing:
- all raw MC arrays
- averaged arrays
- per-run results
"""
function run_model_reduction_experiment_mc(;
    MC_runs::Int=10,
    base_seed::Int=1234,
    savepath::AbstractString="heat2d_model_reduction_results_mc.jld2",
    kwargs...,
)
    @assert MC_runs >= 1 "MC_runs must be at least 1"

    # Run once first to determine array sizes
    rng0 = MersenneTwister(base_seed)
    first_savepath = replace(savepath, ".jld2" => "_run1.jld2")

    first_result = run_model_reduction_experiment(;
        rng=rng0,
        savepath=first_savepath,
        kwargs...
    )

    Js = first_result.Js
    Rs = first_result.Rs
    nJ = length(Js)
    nR = length(Rs)

    # Allocate MC storage
    full_mean_errs = zeros(Float64, nJ, MC_runs)
    full_cov_errs  = zeros(Float64, nJ, MC_runs)
    full_times     = zeros(Float64, nJ, MC_runs)

    red_mean_errs  = zeros(Float64, nJ, nR, MC_runs)
    red_cov_errs   = zeros(Float64, nJ, nR, MC_runs)
    red_times      = zeros(Float64, nJ, nR, MC_runs)

    bt_mean_errs   = zeros(Float64, nR, MC_runs)
    bt_cov_errs    = zeros(Float64, nR, MC_runs)

    # Optional: keep all individual result objects
    all_results = Vector{Any}(undef, MC_runs)
    all_results[1] = first_result

    # Fill first run
    full_mean_errs[:, 1] = first_result.full_mean_err
    full_cov_errs[:, 1]  = first_result.full_cov_err
    full_times[:, 1]     = first_result.full_time

    red_mean_errs[:, :, 1] = first_result.red_mean_err
    red_cov_errs[:, :, 1]  = first_result.red_cov_err
    red_times[:, :, 1]     = first_result.red_time

    bt_mean_errs[:, 1] = first_result.bt_mean_err
    bt_cov_errs[:, 1]  = first_result.bt_cov_err

    # Remaining runs
    prog = Progress(MC_runs; desc="Monte Carlo runs")
    next!(prog)  # first run already completed

    for k in 2:MC_runs
        rngk = MersenneTwister(base_seed + k - 1)
        run_savepath = replace(savepath, ".jld2" => "_run$(k).jld2")

        result_k = run_model_reduction_experiment(;
            rng=rngk,
            savepath=run_savepath,
            kwargs...
        )

        all_results[k] = result_k

        full_mean_errs[:, k] = result_k.full_mean_err
        full_cov_errs[:, k]  = result_k.full_cov_err
        full_times[:, k]     = result_k.full_time

        red_mean_errs[:, :, k] = result_k.red_mean_err
        red_cov_errs[:, :, k]  = result_k.red_cov_err
        red_times[:, :, k]     = result_k.red_time

        bt_mean_errs[:, k] = result_k.bt_mean_err
        bt_cov_errs[:, k]  = result_k.bt_cov_err

        # checkpoint aggregated file
        mc_results = (
            timestamp = string(now()),
            MC_runs = MC_runs,
            base_seed = base_seed,
            Js = Js,
            Rs = Rs,

            full_mean_errs = full_mean_errs,
            full_cov_errs  = full_cov_errs,
            full_times     = full_times,

            red_mean_errs = red_mean_errs,
            red_cov_errs  = red_cov_errs,
            red_times     = red_times,

            bt_mean_errs = bt_mean_errs,
            bt_cov_errs  = bt_cov_errs,

            full_mean_err_avg = vec(mean(full_mean_errs, dims=2)),
            full_cov_err_avg  = vec(mean(full_cov_errs, dims=2)),
            full_time_avg     = vec(mean(full_times, dims=2)),

            red_mean_err_avg = mean(red_mean_errs, dims=3)[:, :, 1],
            red_cov_err_avg  = mean(red_cov_errs, dims=3)[:, :, 1],
            red_time_avg     = mean(red_times, dims=3)[:, :, 1],

            bt_mean_err_avg = vec(mean(bt_mean_errs, dims=2)),
            bt_cov_err_avg  = vec(mean(bt_cov_errs, dims=2)),

            all_results = all_results,
        )

        @save savepath mc_results
        next!(prog)
    end

    # final aggregate
    mc_results = (
        timestamp = string(now()),
        MC_runs = MC_runs,
        base_seed = base_seed,
        Js = Js,
        Rs = Rs,

        full_mean_errs = full_mean_errs,
        full_cov_errs  = full_cov_errs,
        full_times     = full_times,

        red_mean_errs = red_mean_errs,
        red_cov_errs  = red_cov_errs,
        red_times     = red_times,

        bt_mean_errs = bt_mean_errs,
        bt_cov_errs  = bt_cov_errs,

        full_mean_err_avg = vec(mean(full_mean_errs, dims=2)),
        full_cov_err_avg  = vec(mean(full_cov_errs, dims=2)),
        full_time_avg     = vec(mean(full_times, dims=2)),

        red_mean_err_avg = mean(red_mean_errs, dims=3)[:, :, 1],
        red_cov_err_avg  = mean(red_cov_errs, dims=3)[:, :, 1],
        red_time_avg     = mean(red_times, dims=3)[:, :, 1],

        bt_mean_err_avg = vec(mean(bt_mean_errs, dims=2)),
        bt_cov_err_avg  = vec(mean(bt_cov_errs, dims=2)),

        all_results = all_results,
    )

    @save savepath mc_results
    return mc_results
end

##
Js = [500, 1_000, 5_000, 10_000]
Rs = [20, 50, 100, 200, 300]

mc_results = run_model_reduction_experiment_mc(
    MC_runs = 10,
    base_seed = 1234,
    savepath = "data/results_heat2d_bt_ekrmle_mc.jld2",
    Js = Js,
    Rs = Rs,
    steps = 25,
    obs_h = 0.1,
    rel_noise_level = 0.2,
    Nx = 64,
    Ny = 64,
    Δt = 1e-2,
    T_stop = 2.0,
    diffusion_coeffs = 0.5,
)

##

