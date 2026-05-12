using EKRMLE
using LinearAlgebra
using Random
using CairoMakie
using Distributions
using SparseArrays
using ColorSchemes
include("PrettyPlots.jl")


## Setup
N, L = 80, 1.0
obs_ΔN = 8
α = 2.0
τ = 3.0
N_KL = 128
σ₀ = 1.0
d = 32
noise_level = 0.05 # 5% of output
darcy = Darcy_params_2D(N, L, N_KL, obs_ΔN, obs_ΔN, d, α, τ, σ₀)
κ = exp.(darcy.logk_2d)
h = Darcy_2D_solver(darcy, κ)
y_nonoise = get_data(darcy, h)
# create noisy observations
y = copy(y_nonoise)
for i = 1:darcy.n
    noise = rand(Normal(0, noise_level*y[i]))
    y[i] += noise
end
Γ = Array(Diagonal(fill(1.0, length(y))))

## Plot truth
fig, ax = plot_field(darcy, darcy.logk_2d, false)
ax.title = L"\log(a(𝐱;\textbf{v}_{\text{truth}}))"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_truth_d32.pdf",fig)
#save("plots/Darcy_2D_truth.svg",fig)

## Plot solution
fig, ax = plot_field(darcy, h, true)
ax.title = L"p(𝐱)"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_p_manypoints.pdf",fig)
#save("plots/Darcy_2D_p.svg",fig)


## EKRMLE
# Need to solve a Bayesian IP
# Treat prior as 𝒩(0, I)
n, d = length(y), darcy.d
Γ_pr = I(d)
Γ_RLS = Matrix{Float64}(I(n+d))
Γ_RLS[1:n, 1:n] .= Matrix{Float64}(Γ)
Γ_RLS[n+1:end, n+1:end] .= Matrix{Float64}(Γ_pr)
Γ_RLS = Symmetric(Γ_RLS)
y_RLS = vcat(Vector{Float64}(y), zeros(Float64, d))
J = 1000
#v₀ = rand(d, J)
v₀ = rand(MvNormal(vec(zeros(1,d)),Symmetric(Γ_pr)), J)
ekrmleobj = EKRMLEObj(v₀, y_RLS, Γ_RLS)
fwd_RLS_single(darcy, v::AbstractVector) = fwd_RLS_2D(darcy, v)
steps = 25
EKRMLE_run!(ekrmleobj, darcy, fwd_RLS_single, steps)
#EKRMLE_cont_run!(ekrmleobj, darcy, fwd_RLS_single, steps)


## EKI (deterministic)
ekiobj = EKIObj(v₀, y_RLS, Γ_RLS, Γ_RLS)
EKI_run!(ekiobj, darcy, fwd_RLS_single, steps; flavor="vanilla")

## EKI (stochastic)
sekiobj = EKIObj(v₀, y_RLS, Γ_RLS, Γ_RLS)
EKI_run!(sekiobj, darcy, fwd_RLS_single, steps; flavor="stoch")



## Plot EKRMLE field
μ = vec(mean(ekrmleobj.V[end],dims=2))
logk_EKRMLE = get_logk_2D(darcy, μ)
fig, ax = plot_field(darcy, logk_EKRMLE, false)
ax.title = L"\log(a(𝐱;𝐯_{\text{EKRMLE}}))"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_ekrmle_d32.pdf",fig)
#save("plots/Darcy_2D_ekrmle.svg",fig)


## Plot EKI field
μ_EKI = vec(mean(ekiobj.V[end],dims=2))
logk_EKI = get_logk_2D(darcy, μ_EKI)
fig, ax = plot_field(darcy, logk_EKI, false)
ax.title = L"\log(a(𝐱;𝐯_{\text{EKI}}))"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_ekrmle_d32.pdf",fig)
#save("plots/Darcy_2D_ekrmle.svg",fig)

## Plot sEKI field
μ_sEKI = vec(mean(sekiobj.V[end],dims=2))
logk_sEKI = get_logk_2D(darcy, μ_sEKI)
fig, ax = plot_field(darcy, logk_sEKI, false)
ax.title = L"\log(a(𝐱;𝐯_{\text{sEKI}}))"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_ekrmle_d32.pdf",fig)
#save("plots/Darcy_2D_ekrmle.svg",fig)


##

μ_EKRMLE = vec(mean(ekrmleobj.V[end], dims=2))
logk_EKRMLE = get_logk_2D(darcy, μ_EKRMLE)

μ_EKI = vec(mean(ekiobj.V[end], dims=2))
logk_EKI = get_logk_2D(darcy, μ_EKI)

μ_sEKI = vec(mean(sekiobj.V[end], dims=2))
logk_sEKI = get_logk_2D(darcy, μ_sEKI)

# --- Shared color range across all three plots ---
allvals = vcat(vec(darcy.logk_2d), vec(logk_EKRMLE), vec(logk_EKI), vec(logk_sEKI))
crange = extrema(allvals)

# If your field lives on a regular grid, adapt x/y as needed.
# For many Darcy setups, just plotting the matrix directly is enough.
fig = themed_figure(; dark=false, size=(1900, 500)) do fig

    ax0 = Axis(fig[1, 1],
        title = L"\text{truth}",
        titlesize = 50,
    )

    ax1 = Axis(fig[1, 2],
        title = L"\text{EKRMLE}",
        titlesize = 50,
    )

    ax2 = Axis(fig[1, 3],
        title = L"\text{Deterministic EKI}",
        titlesize = 50,
    )

    ax3 = Axis(fig[1, 4],
        title = L"\text{Stochastic EKI}",
        titlesize = 50,
    )

    hm0 = heatmap!(ax0, darcy.logk_2d; colormap=:magma, colorrange=crange)
    hm1 = heatmap!(ax1, logk_EKRMLE; colormap=:magma, colorrange=crange)
    hm2 = heatmap!(ax2, logk_EKI;    colormap=:magma, colorrange=crange)
    hm3 = heatmap!(ax3, logk_sEKI;   colormap=:magma, colorrange=crange)

    hideydecorations!(ax0, grid=false)
    hideydecorations!(ax1, grid=false)
    hideydecorations!(ax2, grid=false)
    hideydecorations!(ax3, grid=false)

    hidexdecorations!(ax0, grid=false)
    hidexdecorations!(ax1, grid=false)
    hidexdecorations!(ax2, grid=false)
    hidexdecorations!(ax3, grid=false)
    
    Colorbar(fig[1, 5], hm0, width=30, ticklabelsize = 25)

    
    colgap!(fig.layout, 20)

    fig
end


## field generated from forward evaluations
logfields = zeros(N,N,J)
logfields_EKI = zeros(N,N,J)
logfields_sEKI = zeros(N,N,J)
for j = 1 : J
    logfields[:,:,j] = get_logk_2D(darcy, ekrmleobj.V[end][:,j])
    logfields_EKI[:,:,j] = get_logk_2D(darcy, ekiobj.V[end][:,j])
    logfields_sEKI[:,:,j] = get_logk_2D(darcy, sekiobj.V[end][:,j])
end

μ_field = mean(logfields, dims=3)

## Uncertainty plot
@inline _colmean(V::AbstractMatrix) = vec(mean(V, dims=2))
function _samplecov(V::AbstractMatrix)
    # computes sample covariance (column-wise)
    J = size(V, 2)
    μ = _colmean(V)
    X = V .- μ
    return (X * X') / (J-1)
end

σ = sqrt.(diag(_samplecov(ekrmleobj.V[end]))) 
upper = μ .+ 2 .* σ
lower = μ .- 2 .* σ

σ_EKI = sqrt.(diag(_samplecov(ekiobj.V[end]))) 
upper_EKI = μ_EKI .+ 2 .* σ_EKI
lower_EKI = μ_EKI .- 2 .* σ_EKI


σ_sEKI = sqrt.(diag(_samplecov(sekiobj.V[end]))) 
upper_sEKI = μ_sEKI .+ 2 .* σ_sEKI
lower_sEKI = μ_sEKI .- 2 .* σ_sEKI
colors = my_custom_dark_theme.palette.color[]

fig = themed_figure(; dark=false, size=(1200, 700)) do fig
    ax1 = Axis(fig[1, 1],
        title     = L"\text{EKRMLE}",
        titlesize = 45,
    )

    band!(ax1, 1:length(μ), lower, upper;
        color = (colors[3], 0.30), 
        label = L"\pm 2\sigma",
    )

    lines!(ax1, darcy.θ_true;
        linewidth   = 10,
        label       = L"\text{truth}",
        color       = colors[4],
    )

    lines!(ax1, μ;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"\text{ensemble mean}",
        color       = colors[3],
    )

    #legend = Legend(fig, ax1; framevisible=false, labelsize=40)

    #fig[1, 2] = legend    # <-- place legend in separate column
    
    # Plot EKI

    ax2 = Axis(fig[2, 1],
        title = L"\text{Deterministic EKI}",
        titlesize = 45,
    )

    band!(ax2, 1:length(μ_EKI), lower_EKI, upper_EKI;
        color = (colors[3], 0.30), 
        label = L"\pm 2\sigma",
    )

    lines!(ax2, darcy.θ_true;
        linewidth   = 10,
        label       = L"𝐯_\text{truth}",
        color       = colors[4],
    )

    lines!(ax2, μ_EKI;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"𝐯_\text{EKI}",
        color       = colors[3],
    )

    #legend = Legend(fig, ax2; framevisible=false, labelsize=40)

    #fig[2, 2] = legend    # <-- place legend in separate column


    # Plot sEKI

    ax3 = Axis(fig[3, 1],
        title = L"\text{Stochastic EKI}",
        titlesize = 45,
    )

    band!(ax3, 1:length(μ_sEKI), lower_sEKI, upper_sEKI;
        color = (colors[3], 0.30), 
        label = L"\pm 2\sigma",
    )

    lines!(ax3, darcy.θ_true;
        linewidth   = 10,
        label       = L"𝐯_\text{truth}",
        color       = colors[4],
    )

    lines!(ax3, μ_sEKI;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"𝐯_\text{sEKI}",
        color       = colors[3],
    )

    #legend = Legend(fig, ax3; framevisible=false, labelsize=40)

    #fig[3, 2] = legend    # <-- place legend in separate column


    #colsize!(fig.layout, 2, Auto())   # auto-size legend column
    hidexdecorations!(ax1, grid=false)
    hidexdecorations!(ax2, grid=false)
    Legend(fig[end+1, :], ax1, orientation=:horizontal,
                             labelsize=45,
                             patchsize = (80,40),
                             framevisible=false)
    #axislegend(ax1; position=:rb, framevisible=false, labelsize=35)
    fig   # important: return the figure from the do-block
end


#save("plots/Darcy_2D_d32_uncertain.pdf",fig)

## Visualize a slice with uncertainty

ℓ = 34 # pick slice index
EKRMLE_slices = logfields[:,ℓ,:]
μ_slice = _colmean(EKRMLE_slices)
C = sqrt.(diag(_samplecov(EKRMLE_slices)))
truth_slice = darcy.logk_2d[:, ℓ]
upper_slice = μ_slice .+ 2 .* C
lower_slice = μ_slice .- 2 .* C

EKI_slices = logfields_EKI[:,ℓ,:]
μ_slice_EKI = _colmean(EKI_slices)
C_EKI = sqrt.(diag(_samplecov(EKI_slices)))
upper_slice_EKI = μ_slice_EKI .+ 2 .* C_EKI
lower_slice_EKI = μ_slice_EKI .- 2 .* C_EKI

sEKI_slices = logfields_sEKI[:,ℓ,:]
μ_slice_sEKI = _colmean(sEKI_slices)
C_sEKI = sqrt.(diag(_samplecov(sEKI_slices)))
upper_slice_sEKI = μ_slice_sEKI .+ 2 .* C_sEKI
lower_slice_sEKI = μ_slice_sEKI .- 2 .* C_sEKI

colors = my_custom_dark_theme.palette.color[]

fig = themed_figure(; dark=false, size=(1300, 800)) do fig
    ax1 = Axis(fig[1, 1],
        title     = L"\text{EKRMLE}",
        titlesize = 45,
    )

    band!(ax1, 1:length(μ_slice), lower_slice, upper_slice;
        color = (colors[3], 0.30), 
        label = L"\pm 2\sigma",
    )

    lines!(ax1, truth_slice;
        linewidth   = 10,
        label       = L"\text{true permeability}",
        color       = colors[4],
    )

    lines!(ax1, μ_slice;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"\text{ensemble mean}",
        color       = colors[3],
    )

    #legend = Legend(fig, ax1; framevisible=false, labelsize=50)

    #fig[1, 2] = legend    # <-- place legend in separate column

    # Plot EKI

    ax2 = Axis(fig[2, 1],
        title = L"\text{Deterministic EKI}",
        titlesize = 45,
    )

    band!(ax2, 1:length(μ_slice_EKI), lower_slice_EKI, upper_slice_EKI;
        color = (colors[3], 0.30), 
        label = L"\pm 2\sigma",
    )

    lines!(ax2, truth_slice;
        linewidth   = 10,
        label       = L"\text{truth}",
        color       = colors[4],
    )

    lines!(ax2, μ_slice_EKI;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"\text{EKI}",
        color       = colors[3],
    )

    #legend = Legend(fig, ax2; framevisible=false, labelsize=50)

    #fig[2, 2] = legend  


    # Plot sEKI

    ax3 = Axis(fig[3, 1],
        title = L"\text{Stochastic EKI}",
        titlesize = 45,
    )

    band!(ax3, 1:length(μ_slice_sEKI), lower_slice_sEKI, upper_slice_sEKI;
        color = (colors[3], 0.30), 
        label = L"\pm 2\sigma",
    )

    lines!(ax3, truth_slice;
        linewidth   = 10,
        label       = L"\text{truth}",
        color       = colors[4],
    )

    lines!(ax3, μ_slice_sEKI;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"\text{sEKI}",
        color       = colors[3],
    )

    #legend = Legend(fig, ax3; framevisible=false, labelsize=50)

    #fig[3, 2] = legend    # <-- place legend in separate column



    #colsize!(fig.layout, 2, Auto())   # auto-size legend column
    hidexdecorations!(ax1, grid=false)
    hidexdecorations!(ax2, grid=false)
    #axislegend(ax1; position=:rb, framevisible=false, labelsize=35)
    Legend(fig[end+1, :], ax1, orientation=:horizontal,
                             labelsize=45,
                             patchsize = (80,40),
                             framevisible=false)
    fig   # important: return the figure from the do-block
end
display(fig)
#save("plots/Darcy_2D_unc.png",fig)



## All lines together
fig = themed_figure(; dark=false, size=(1200, 350)) do fig
    ax = Axis(fig[1, 1],
        title     = L"\text{Permeability slice comparison}",
        titlesize = 40,
    )

    x = 1:length(μ_slice)

    # EKRMLE uncertainty band only
    band!(ax, x, lower_slice, upper_slice;
        color = (colors[3], 0.25),
        label = L"\text{EKRMLE } \pm 2\sigma",
    )

    # Truth
    lines!(ax, x, truth_slice;
        linewidth = 10,
        linestyle = :solid,
        label = L"\text{true permeability}",
        color = colors[4],
    )

    # EKRMLE mean
    lines!(ax, x, μ_slice;
        linewidth = 10,
        linestyle = :dash,
        label = L"\text{EKRMLE}",
        color = colors[3],
    )

    # Deterministic EKI mean
    lines!(ax, x, μ_slice_EKI;
        linewidth = 8,
        linestyle = :dot,
        label = L"\text{Deterministic EKI}",
        color = colors[1],
    )

    # Stochastic EKI mean
    lines!(ax, x, μ_slice_sEKI;
        linewidth = 6,
        linestyle = :dashdot,
        label = L"\text{Stochastic EKI}",
        color = colors[2],
    )

    axislegend(ax;
        position = :rt,
        framevisible = false,
        labelsize = 25,
        patchsize = (70, 20),
    )

    fig
end

display(fig)
#save("plots/Darcy_2D_uncertain.pdf",fig)