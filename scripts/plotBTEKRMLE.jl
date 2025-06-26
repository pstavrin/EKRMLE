using EKRMLE
using CairoMakie
using ControlSystems
using LinearAlgebra
using Distributions
using ProgressMeter
using LaTeXStrings
using MAT
using ColorSchemes

## Load data
data = matread("data/BTEKI.mat")
for (k, v) in data
    @eval $(Symbol(k)) = $v
end
rs = length(Rs)
js = length(Js)


## Plot full runs
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=rs+4)]
colors = colors[end:-1:1]
fig = Figure(size=(1800,600))
ax1 = Axis(fig[1,1],
    xlabel=L"ensemble size $J$",
    title=L"\text{Posterior mean error}",
    yscale=log10,
    xscale=log10,
    xlabelsize = 38,
    ylabelsize = 38,
    titlesize = 40,
    xticklabelsize = 38,
    yticklabelsize = 38
)
#X = range(0,iters,length = iters+1)
for j = 1 : rs
    r = Rs[j]
    lines!(ax1, Js, vec(mean(mu_BTEKI_errs,dims=3)[:,j]),
     linewidth=10,
     color = colors[j+1],
     label = latexstring("r = $(r)"))
end
#scatterlines!(ax1, Rs, vec(μ_BTEKI_errs[4,:]), label = L"J = 100")
lines!(ax1, Js, vec(mean(mu_EKI_errs,dims=3)),
    linewidth = 10, color = :black, linestyle=:dash,
    label = L"\text{Full model}"
)
scatter!(
    ax1, Js, vec(mean(mu_EKI_errs,dims=3)),
    color=:black, markersize=30,
    marker =:circle,
    label = L"\text{Full model}"
)
#axislegend(ax1,position =:lt, labelsize = 38,merge=true,framevisible=false)
#display(fig)
#save(joinpath("plots", "meanVSparticles.pdf"), fig)

#fig = Figure(size=(1000,500))
ax2 = Axis(fig[1,2],
    xlabel=L"ensemble size $J$",
    title=L"\text{Posterior covariance error}",
    yscale=log10,
    xscale=log10,
    xlabelsize = 38,
    ylabelsize = 0,
    titlesize = 40,
    xticklabelsize = 38,
    yticklabelsize = 0
)
#X = range(0,iters,length = iters+1)
for j = 1 : rs
    r = Rs[j]
    lines!(ax2, Js, vec(mean(Gamma_BTEKI_errs,dims=3)[:,j]),
     linewidth=10,
     color = colors[j+1],
     label = latexstring("r = $(r)"))
end
#scatterlines!(ax1, Rs, vec(μ_BTEKI_errs[4,:]), label = L"J = 100")
lines!(ax2, Js, vec(mean(Gamma_EKI_errs,dims=3)),
    linewidth = 10, color = :black, linestyle=:dash,
    label = L"\text{Full model}"
)
scatter!(
    ax2, Js, vec(mean(Gamma_EKI_errs,dims=3)),
    color=:black, markersize=30,
    marker =:circle,
    label = L"\text{Full model}"
)
axislegend(ax2,position =:lb, labelsize = 38,
merge=true,framevisible=false)
linkyaxes!(ax1,ax2)
display(fig)

#save(joinpath("plots", "VSparticles.pdf"), fig)

## Plot BT EKI runs
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=js+2)]
colors = colors[end:-1:1]
fig = Figure(size=(1800,600))
ax1 = Axis(fig[1,1],
    xlabel=L"reduced model size $r$",
    title=L"\text{Posterior mean error}",
    yscale=log10,
    xlabelsize = 38,
    ylabelsize = 38,
    titlesize = 40,
    xticklabelsize = 38,
    yticklabelsize = 38
)
for j = 1 : js
    J = Js[j]
    lines!(ax1, Rs, vec(mean(mu_BTEKI_errs,dims=3)[j,:]),
     linewidth=10,
     color = colors[j+1],
     label = latexstring("J = $(J)"))
end
#scatterlines!(ax1, Rs, vec(μ_BTEKI_errs[4,:]), label = L"J = 100")
lines!(ax1, Rs, vec(mean(mu_BT_errs,dims=3)[1,:]),
    linewidth = 10, color = :black, linestyle=:dash,
    label = L"\text{Exact}"
)
scatter!(
    ax1, Rs, vec(mean(mu_BT_errs,dims=3)[1,:]),
    color=:black, markersize=30,
    marker =:circle,
    label = L"\text{Exact}"
)
#axislegend(ax2,position =:lb, labelsize = 25,merge=true,framevisible=false)
#display(fig)
#save(joinpath("plots", "meanVSbasis.pdf"), fig)

#fig = Figure(size=(750,500))
ax2 = Axis(fig[1,2],
    xlabel=L"reduced model size $r$",
    title=L"\text{Posterior covariance error}",
    yscale=log10,
    xlabelsize = 38,
    ylabelsize = 0,
    titlesize = 40,
    xticklabelsize = 38,
    yticklabelsize = 0
)
for j = 1 : js
    J = Js[j]
    lines!(ax2, Rs, vec(mean(Gamma_BTEKI_errs,dims=3)[j,:]),
     linewidth=10,
     color = colors[j+1],
     label = latexstring("J = $(J)"))
end
#scatterlines!(ax1, Rs, vec(μ_BTEKI_errs[4,:]), label = L"J = 100")
lines!(ax2, Rs, vec(mean(Gamma_BT_errs,dims=3)[1,:]),
    linewidth = 10, color = :black, linestyle=:dash,
    label = L"\text{Exact}"
)
scatter!(
    ax2, Rs, vec(mean(Gamma_BT_errs,dims=3)[1,:]),
    color=:black, markersize=30,
    marker =:circle,
    label = L"\text{Exact}"
)
axislegend(ax2,position =:lb, labelsize = 38,
merge=true,framevisible=false)
linkyaxes!(ax1,ax2)
display(fig)
#save(joinpath("plots", "VSbasis.pdf"), fig)
