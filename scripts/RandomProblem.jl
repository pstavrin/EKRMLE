using EKRMLE
using LinearAlgebra
using Random
using CairoMakie
using Distributions
using ColorSchemes
include("PrettyPlots.jl")


@inline colmean(V::AbstractMatrix) = vec(mean(V, dims=2))

function _samplecov(V::AbstractMatrix)
    # computes sample covariance (column-wise)
    J = size(V, 2)
    μ = colmean(V)
    X = V .- μ
    return (X * X') / (J-1)
end

## Setup
n = 100
d = 35
J = 10000
prob = randomLinearProblemObj(n, d, J; rankdef=false)


## EKRMLE
obj = EKRMLEObj(prob.V0, prob.y, prob.Γ) # create EKRMLE object
iters = 100
H_s(prob::randomLinearProblemObj, v::AbstractVector) = prob.H * v
EKRMLE_run!(obj, prob, H_s, iters)

##
C = _samplecov(obj.V[end])
## 
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1], title=L"\text{Cov}[\textbf{v}_\text{end}^{(1:J)}]", titlesize=30)
ax1.yreversed=true
ax2 = Axis(fig[1, 3], title = L"(\textbf{H}^⊤\textbf{Γ}^{-1}\textbf{H})^†", titlesize=30)
ax2.yreversed=true
hm1 = heatmap!(ax1, C; colormap=:magma)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(ax2, prob.pHess; colormap=:magma)
Colorbar(fig[1, 4], hm2)
display(fig)
#save("plots/cov_compare.pdf", fig)


## Spectral projectors
projectors = spectralproj(prob, _samplecov(prob.V0))
P = real.(projectors.P)

## 
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1], title=L"\text{Cov}[\textbf{Pv}_\text{end}^{(1:J)}]", titlesize=30)
ax1.yreversed=true
ax2 = Axis(fig[1, 3] , title = L"\textbf{P}(\textbf{H}^⊤\textbf{Γ}^{-1}\textbf{H})^†\textbf{P}^⊤", titlesize=30)
ax2.yreversed=true
hm1 = heatmap!(ax1, P*C*P'; colormap=:magma)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(ax2, P*prob.pHess*P'; colormap=:magma)
Colorbar(fig[1, 4], hm2)
display(fig)
#save("plots/cov_compare_P.pdf", fig)

## plot some marginals
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=5)]
m1 = 2
m2 = 6
V_marg = obj.V[end][[m1,m2],:]
PV_marg = (P*obj.V[end])[[m1,m2],:]

fig = Figure(size=(900,600))
ax = Axis(fig[1,1], xlabel="", ylabel="", title="marginals")
scatter!(ax, V_marg[1,:], V_marg[2,:]; markersize=15, label="true", color=(colors[2], 0.30))
scatter!(ax, PV_marg[1,:], PV_marg[2,:]; markersize=15 ,label = "P-space", color=(colors[4], 0.25))

axislegend(ax; position=:rb, framevisible = false, labelsize=20)

display(fig)


##
μ = colmean(obj.V[end])
Pμ = P*μ
fig = Figure(size=(1000,500))
ax1 = Axis(fig[1, 1], title = L"\text{Full state space }\mathbb{R}^d", titlesize=30)
ax2 = Axis(fig[2,1], title = L"\textbf{P}\text{ space}", titlesize=30)
scatterlines!(ax1, prob.v_star;linewidth=5, markersize=15, label=L"\textbf{v}^\star", color=colors[2])
scatterlines!(ax1, μ;linewidth=5, linestyle=:dash, markersize=15, label=L"\text{E}[\textbf{v}_\text{end}^{(1:J)}]", color=colors[4])
scatterlines!(ax2, P*prob.v_star;linewidth=5, markersize=20, label=L"\textbf{Pv}^\star", color=colors[2])
scatterlines!(ax2, Pμ;linestyle=:dash, linewidth=4, marker=:cross, markersize=18, label=L"\text{E}[\textbf{Pv}_\text{end}^{(1:J)}]", color=colors[4])
axislegend(ax1; position=:lb, framevisible = false, labelsize=30)
axislegend(ax2; position=:lb, framevisible = false, labelsize=30)
display(fig)
#save("plots/mean_compare.pdf", fig)


## PrettyPlot
colors = my_custom_dark_theme.palette.color[]

fig = themed_figure(; dark=true, size=(1000, 500)) do fig
    ax1 = Axis(fig[1, 1],
        title     = L"\text{Full state space }\mathbb{R}^d",
        titlesize = 30,
    )
    ax2 = Axis(fig[2, 1],
        title     = L"\textbf{P}\text{ space}",
        titlesize = 30,
    )

    scatterlines!(ax1, prob.v_star;
        linewidth   = 7,
        markersize  = 15,
        label       = L"\textbf{v}^\star",
        color       = colors[2],
    )

    scatterlines!(ax1, μ;
        linewidth   = 6,
        linestyle   = :dash,
        markersize  = 15,
        label       = L"\text{E}[\textbf{v}_\text{end}^{(1:J)}]",
        color       = colors[4],
    )

    scatterlines!(ax2, P * prob.v_star;
        linewidth   = 7,
        markersize  = 20,
        label       = L"\textbf{Pv}^\star",
        color       = colors[2],
    )

    scatterlines!(ax2, Pμ;
        linestyle   = :dash,
        linewidth   = 6,
        marker      = :cross,
        markersize  = 18,
        label       = L"\text{E}[\textbf{Pv}_\text{end}^{(1:J)}]",
        color       = colors[4],
    )

    axislegend(ax1; position=:lb, framevisible=false, labelsize=30)
    axislegend(ax2; position=:rt, framevisible=false, labelsize=30)

    fig   # important: return the figure from the do-block
end



## compare against black box sampler
bb_ens = rand(MvNormal(P*prob.v_star, Symmetric(P*prob.pHess*P' + 1e-15*I)), J)
m1 = 21
m2 = 11

bb_marg = bb_ens[[m1, m2], :]
PV_marg = (P * obj.V[end])[[m1, m2], :]

colors = my_custom_dark_theme.palette.color[]  # your 25-color palette

fig = themed_figure(; dark=true, size=(900, 500)) do fig
    ax = Axis(fig[1, 1],
        xlabel = "",
        ylabel = "",
        title  = L"\text{RMLE in }𝐏\text{ space}",
        titlesize = 35,
    )

    scatter!(ax, bb_marg[1, :], bb_marg[2, :];
        markersize = 20,
        label      = L"\text{black box}",
        color      = (colors[2], 0.30),   # same as before, now under your theme
    )

    scatter!(ax, PV_marg[1, :], PV_marg[2, :];
        markersize = 20,
        label      = L"\text{EKRMLE}",
        color      = (colors[4], 0.30),
    )

    axislegend(ax; position=:lb, framevisible=false, labelsize=30)

    fig
end

display(fig)