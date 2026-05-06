using EnsembleKalmanRMLE
using LinearAlgebra
using Random
using CairoMakie
using Distributions
using SparseArrays
using ColorSchemes
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses
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
#ekrmleobj = EKRMLEContObj(v₀, y_RLS, Γ_RLS)
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

display(fig)
save("plots/Darcy_2D_sbs.pdf",fig)

## Same figure but in dark mode

fig = themed_figure(; dark=true, size=(1200, 600)) do fig

    ax0 = Axis(fig[1, 1],
        title = L"\text{truth}",
        titlesize = 50,
    )

    ax1 = Axis(fig[1, 2],
        title = L"\text{EKRMLE}",
        titlesize = 50,
    )
#=
    ax2 = Axis(fig[1, 3],
        title = L"\text{Deterministic EKI}",
        titlesize = 50,
    )
        

    ax3 = Axis(fig[1, 4],
        title = L"\text{Stochastic EKI}",
        titlesize = 50,
    )
        =#

    hm0 = heatmap!(ax0, darcy.logk_2d; colormap=:magma, colorrange=crange)
    hm1 = heatmap!(ax1, logk_EKRMLE; colormap=:magma, colorrange=crange)
    #hm2 = heatmap!(ax2, logk_EKI;    colormap=:magma, colorrange=crange)
    #hm3 = heatmap!(ax3, logk_sEKI;   colormap=:magma, colorrange=crange)

    hideydecorations!(ax0, grid=false)
    hideydecorations!(ax1, grid=false)
    #hideydecorations!(ax2, grid=false)
    #hideydecorations!(ax3, grid=false)

    hidexdecorations!(ax0, grid=false)
    hidexdecorations!(ax1, grid=false)
    #hidexdecorations!(ax2, grid=false)
    #hidexdecorations!(ax3, grid=false)
    
    Colorbar(fig[1, 3], hm0, width=35, ticklabelsize = 35,
                         labelcolor=:white,
                         ticklabelcolor=:white,
                         tickcolor=:white)
    
    
    ax0.bottomspinecolor=:black
    ax0.leftspinecolor=:black
    ax1.bottomspinecolor=:black
    ax1.leftspinecolor=:black
    #ax2.bottomspinecolor=:black
    #ax2.leftspinecolor=:black
    #ax3.bottomspinecolor=:black
    #ax3.leftspinecolor=:black
 
    
    colgap!(fig.layout, 20)

    fig
end

display(fig)

#save("plots/Darcy_2D_sol_comp_d32.svg",fig)


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

## Compare with ground truth
plot_field_sbs(darcy, darcy.logk_2d, logk_EKRMLE)

## Compare with ground truth
plot_field_sbs(darcy, darcy.logk_2d, μ_field[:,:])
## Plot error
fig, ax = plot_field(darcy, abs.(darcy.logk_2d - logk_EKRMLE))
ax.title = L"\text{Absolute error}"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_error_d32.pdf",fig)
#save("plots/Darcy_2D_error.svg",fig)


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


## Same figure but in dark mode

fig = themed_figure(; dark=true, size=(1200, 250)) do fig
    ax1 = Axis(fig[1, 1],
        #title     = L"\text{EKRMLE}",
        titlesize = 45,
        xtickcolor = :white,
        ytickcolor = :white,
    )

    band!(ax1, 1:length(μ), lower, upper;
        color = (colors[5], 0.40), 
        label = L"\pm 2\sigma",
    )

    lines!(ax1, darcy.θ_true;
        linewidth   = 10,
        label       = L"\text{true }𝐯",
        color       = colors[2],
    )

    lines!(ax1, μ;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"\text{ensemble mean}",
        color       = colors[5],
    )

    #legend = Legend(fig, ax1; framevisible=false, labelsize=40)

    #fig[1, 2] = legend    # <-- place legend in separate column
    
    # Plot EKI
    #=
    ax2 = Axis(fig[2, 1],
        title = L"\text{Deterministic EKI}",
        titlesize = 45,
        xtickcolor = :white,
        ytickcolor = :white,
    )

    band!(ax2, 1:length(μ_EKI), lower_EKI, upper_EKI;
        color = (colors[5], 0.40), 
        label = L"\pm 2\sigma",
    )

    lines!(ax2, darcy.θ_true;
        linewidth   = 10,
        label       = L"𝐯_\text{truth}",
        color       = colors[2],
    )

    lines!(ax2, μ_EKI;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"𝐯_\text{EKI}",
        color       = colors[5],
    )

    #legend = Legend(fig, ax2; framevisible=false, labelsize=40)

    #fig[2, 2] = legend    # <-- place legend in separate column


    # Plot sEKI

    ax3 = Axis(fig[3, 1],
        title = L"\text{Stochastic EKI}",
        titlesize = 45,
        xtickcolor = :white,
        ytickcolor = :white,
    )

    band!(ax3, 1:length(μ_sEKI), lower_sEKI, upper_sEKI;
        color = (colors[5], 0.40), 
        label = L"\pm 2\sigma",
    )

    lines!(ax3, darcy.θ_true;
        linewidth   = 10,
        label       = L"𝐯_\text{truth}",
        color       = colors[2],
    )

    lines!(ax3, μ_sEKI;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"𝐯_\text{sEKI}",
        color       = colors[5],
    )

    #legend = Legend(fig, ax3; framevisible=false, labelsize=40)

    #fig[3, 2] = legend    # <-- place legend in separate column

    =#
    #colsize!(fig.layout, 2, Auto())   # auto-size legend column
    hidexdecorations!(ax1, grid=false)
    #hidexdecorations!(ax2, grid=false)
    Legend(fig[end+1, :], ax1, orientation=:horizontal,
                             labelsize=45,
                             patchsize = (80,40),
                             framevisible=false)
    #axislegend(ax1; position=:rb, framevisible=false, labelsize=35)
    fig   # important: return the figure from the do-block
end

display(fig)

#save("plots/Darcy_2D_d32_uncertain.svg",fig)



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

## Same figure but in dark mode

fig = themed_figure(; dark=true, size=(1200, 700)) do fig
    ax1 = Axis(fig[1, 1],
        title     = L"\text{EKRMLE}",
        titlesize = 45,
        xtickcolor = :white,
        ytickcolor = :white,
    )

    band!(ax1, 1:length(μ_slice), lower_slice, upper_slice;
        color = (colors[5], 0.40), 
        label = L"\pm 2\sigma",
    )

    lines!(ax1, truth_slice;
        linewidth   = 10,
        label       = L"\text{true permeability}",
        color       = colors[2],
    )

    lines!(ax1, μ_slice;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"\text{ensemble mean}",
        color       = colors[5],
    )

    #legend = Legend(fig, ax1; framevisible=false, labelsize=50)

    #fig[1, 2] = legend    # <-- place legend in separate column

    # Plot EKI

    ax2 = Axis(fig[2, 1],
        title = L"\text{Deterministic EKI}",
        titlesize = 45,
        xtickcolor = :white,
        ytickcolor = :white,
    )

    band!(ax2, 1:length(μ_slice_EKI), lower_slice_EKI, upper_slice_EKI;
        color = (colors[5], 0.40), 
        label = L"\pm 2\sigma",
    )

    lines!(ax2, truth_slice;
        linewidth   = 10,
        label       = L"\text{truth}",
        color       = colors[2],
    )

    lines!(ax2, μ_slice_EKI;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"\text{EKI}",
        color       = colors[5],
    )

    #legend = Legend(fig, ax2; framevisible=false, labelsize=50)

    #fig[2, 2] = legend  


    # Plot sEKI

    ax3 = Axis(fig[3, 1],
        title = L"\text{Stochastic EKI}",
        titlesize = 45,
        xtickcolor = :white,
        ytickcolor = :white,
    )

    band!(ax3, 1:length(μ_slice_sEKI), lower_slice_sEKI, upper_slice_sEKI;
        color = (colors[5], 0.40), 
        label = L"\pm 2\sigma",
    )

    lines!(ax3, truth_slice;
        linewidth   = 10,
        label       = L"\text{truth}",
        color       = colors[2],
    )

    lines!(ax3, μ_slice_sEKI;
        linewidth   = 10,
        linestyle   = :dash,
        label       = L"\text{sEKI}",
        color       = colors[5],
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
save("plots/Darcy_2D_uncertain.pdf",fig)

## Surface plot EKRMLE
μ_surface     = dropdims(mean(logfields; dims=3); dims=3)
σ_surface     = dropdims(std(logfields; dims=3, corrected=true); dims=3)

upper_surface = μ_surface .+ 2 .* σ_surface
lower_surface = μ_surface .- 2 .* σ_surface


X = 1:N
Y = 1:N
fig = themed_figure(; dark=true, size=(2400, 1400)) do fig
    ax1 = Axis3(fig[1, 1],
        #title     = L"\text{EKRMLE}",
        titlesize = 45,

        xlabel = L"𝐱_1",
        ylabel = L"𝐱_2",
        zlabel = L"\log\, a(𝐱;𝐯)",

        xlabelsize = 80,
        ylabelsize = 80,
        zlabelsize = 80,

        # axis labels
        xlabelcolor = :white,
        ylabelcolor = :white,
        zlabelcolor = :white,
        titlecolor  = :white,

        # tick labels
        xticklabelcolor = :white,
        yticklabelcolor = :white,
        zticklabelcolor = :white,

        # tick marks
        xtickcolor = :white,
        ytickcolor = :white,
        ztickcolor = :white,

        # grid lines
        xgridcolor = (:white,0.3),
        ygridcolor = (:white,0.3),
        zgridcolor = (:white,0.3),

        xspinecolor_1 = (:white,0.3),
        xspinecolor_2 = (:white,0.3),
        xspinecolor_3 = (:white,0.3),
        xspinecolor_4 = (:white,0.3),

        yspinecolor_1 = (:white,0.3),
        yspinecolor_2 = (:white,0.3),
        yspinecolor_3 = (:white,0.3),
        yspinecolor_4 = (:white,0.3),

        zspinecolor_1 = (:white,0.3),
        zspinecolor_2 = (:white,0.3),
        zspinecolor_3 = (:white,0.3),
        zspinecolor_4 = (:white,0.3),

        )


        ax1.azimuth = -1
        ax1.elevation = 0.1

    #s1 = surface!(ax1, X, Y, logk_EKRMLE; color=colors[2])

    
    s4 = surface!(ax1, X, Y, lower_surface, color=colors[5],
                            transparency = true,
                            alpha = 0.4)   


    s2 = surface!(ax1, X, Y, darcy.logk_2d, color=colors[2],
                            transparency = false,
                            alpha = 1,
                            label = L"\text{truth}")      
                            
    s3 = surface!(ax1, X, Y, upper_surface, color=colors[5],
                            transparency = true,
                            alpha = 0.4,
                            label = L"\pm 2σ")


    #legend = Legend(fig, ax1; framevisible=false, labelsize=50)

    #fig[1, 2] = legend    # <-- place legend in separate column

    legend_elements = [
        PolyElement(color = colors[2]),                     # Truth
        PolyElement(color = (colors[5], 0.4))               # Uncertainty band
    ]

    Legend(fig[2, 1],
        legend_elements,
        [L"\text{truth}", L"\pm 2\sigma"],
        orientation = :horizontal,
        framevisible = false,
        labelsize = 80,
        patchsize = (100,70)
    )



    fig   # important: return the figure from the do-block
end

display(fig)

##
#save("plots/Darcy_2D_surfacecomp_k256_d32_view1.png",fig)


##
nframes = 360

record(fig, "plots/rotating_surface_d32.mp4", 1:nframes; framerate=30) do i
    ax = content(fig[1, 1])
    ax.azimuth = 2*π * (i - 1) / nframes
end


## Surface plot EKI
μ_surface_EKI     = dropdims(mean(logfields_EKI; dims=3); dims=3)
σ_surface_EKI     = dropdims(std(logfields_EKI; dims=3, corrected=true); dims=3)

upper_surface_EKI = μ_surface_EKI .+ 2 .* σ_surface_EKI
lower_surface_EKI = μ_surface_EKI .- 2 .* σ_surface_EKI


X = 1:N
Y = 1:N
fig = themed_figure(; dark=true, size=(1200, 700)) do fig
    ax1 = Axis3(fig[1, 1],
        title     = L"\text{Deterministic EKI}",
        titlesize = 45,

        # axis labels
        xlabelcolor = :white,
        ylabelcolor = :white,
        zlabelcolor = :white,
        titlecolor  = :white,

        # tick labels
        xticklabelcolor = :white,
        yticklabelcolor = :white,
        zticklabelcolor = :white,

        # tick marks
        xtickcolor = :white,
        ytickcolor = :white,
        ztickcolor = :white,

        # grid lines
        xgridcolor = (:white,0.3),
        ygridcolor = (:white,0.3),
        zgridcolor = (:white,0.3),

        xspinecolor_1 = (:white,0.3),
        xspinecolor_2 = (:white,0.3),
        xspinecolor_3 = (:white,0.3),
        xspinecolor_4 = (:white,0.3),

        yspinecolor_1 = (:white,0.3),
        yspinecolor_2 = (:white,0.3),
        yspinecolor_3 = (:white,0.3),
        yspinecolor_4 = (:white,0.3),

        zspinecolor_1 = (:white,0.3),
        zspinecolor_2 = (:white,0.3),
        zspinecolor_3 = (:white,0.3),
        zspinecolor_4 = (:white,0.3),

        )


        ax1.azimuth = 1.5
        ax1.elevation = 0.1


    #s1 = surface!(ax1, X, Y, logk_EKRMLE; color=colors[2])

    
    #s4 = surface!(ax1, X, Y, lower_surface_EKI, color=colors[5],
     #                       transparency = true,
      #                      alpha = 0.4)   


    s2 = surface!(ax1, X, Y, darcy.logk_2d, color=colors[2],
                            transparency = false,
                            alpha = 1)      
                            
    s3 = surface!(ax1, X, Y, μ_surface_EKI, color=colors[5],
                            transparency = true,
                            alpha = 0.4)


    #legend = Legend(fig, ax1; framevisible=false, labelsize=50)

    #fig[1, 2] = legend    # <-- place legend in separate column


    fig   # important: return the figure from the do-block
end

display(fig)

##
save("plots/Darcy_2D_surfacecomp_EKI_k128_d32_view2.png",fig)


## Interactive surface plot

using GLMakie
GLMakie.activate!()
colors = parse.(Colorant, colors)

fig = Figure(size = (1200, 700))

ax1 = Axis3(fig[1, 1],
    title = L"\text{EKRMLE}",
    titlesize = 45,
)

ax1.azimuth = 1.5
ax1.elevation = 0.1

Clower = fill(colors[3], size(lower_surface))
Ctrue  = fill(colors[4], size(darcy.logk_2d))
Cupper = fill(colors[3], size(upper_surface))

surface!(ax1, X, Y, lower_surface; color = Clower, transparency=true, alpha = 0.5)
surface!(ax1, X, Y, darcy.logk_2d; color = Ctrue)
surface!(ax1, X, Y, upper_surface; color = Cupper, transparency=true, alpha = 0.5)

display(fig)
CairoMakie.activate!()


## EKS
dist = Parameterized(MvNormal(vec(zeros(1, d)), Symmetric(Γ_pr)))
constraint = repeat([no_constraint()], d)
prior = ParameterDistribution(dist, constraint, "v")
init   = EKP.construct_initial_ensemble(prior, J)
eksobj = EKP.EnsembleKalmanProcess(init, y, Γ, Sampler(prior))
for n in 1:steps
    θ_ens = EKP.get_ϕ_final(prior, eksobj)            # d × J
    G_ens = [fwd_2D(darcy, θ_ens[:, j]) for j in 1:J]
    g_ens = hcat(G_ens...)                            # n × J
    EKP.update_ensemble!(eksobj, g_ens)
end


## EKS log field
μ_EKS = vec(mean(EKP.get_ϕ_final(prior, eksobj),dims=2))
logk_EKS = get_logk_2D(darcy, μ_EKS)
fig, ax = plot_field(darcy, logk_EKS, false)
ax.title = L"\log(a(𝐰; 𝐯_{\text{EKS}}))"
ax.titlesize = 40
display(fig)
save("plots/Darcy_2D_eks.svg",fig)

##
plot_field_sbs(darcy, logk_EKRMLE, logk_EKS; titles=("EKRMLE", "EKS"))

## EKS error
fig, ax = plot_field(darcy, abs.(darcy.logk_2d - logk_EKS))
ax.title = L"\text{Absolute error EKS}"
ax.titlesize = 40
display(fig)
#save("plots/Darcy_2D_error.pdf",fig)
#save("plots/Darcy_2D_error.svg",fig)



## Plot some marginals
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=5)]
m1 = 2
m2 = 5
V_marg = ekrmleobj.V[end][[m1,m2],:]
EKS_marg = EKP.get_ϕ_final(prior, eksobj)[[m1,m2],:]

fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="", ylabel="", title="marginals")
scatter!(ax, V_marg[1,:], V_marg[2,:]; markersize=15, label="EKRMLE", color=(colors[2], 0.50))
scatter!(ax, EKS_marg[1,:], EKS_marg[2,:]; markersize=15 ,label = "EKS", color=(colors[4], 0.50))

axislegend(ax; position=:rb, framevisible = false, labelsize=20)

display(fig)