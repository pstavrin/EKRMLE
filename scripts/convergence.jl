using EKRMLE
using CairoMakie
using LinearAlgebra
using Statistics
using ColorSchemes
using LaTeXStrings
## Setup
n = 500 # observation space dimension
d = 1000 # state space dimension
J = 10000 # ensemble size
Jsmall = 10 # small ensemble size
iters = 100 # algorithmic iterations for large ensemble
iters_sm = 100 # iterations for small ensemble
H,Σ,v₀ = randomProblem(n,d,J) # generate random problem
v₀small = v₀[:,1:Jsmall] # small ensemble
truth = rand(d,1) # ground truth
ε = generate_Gaussian_noise(1, Σ) # noise for measurements
y = H*truth + ε # noisy data
H⁺ = pinv(H'*(Σ\H))*((H')/Σ) # weighted pseudoinverse
Hₚ = pinv(H'*(Σ\H))
v_star = H⁺*y # least squares solution

## Iterate EKI
V,Γ,ỹ = EKR(y,H,J,v₀,Σ,iters+1; method="adjfree",store_ens=true)
Vsmall,Γsmall,ỹsmall = EKR(y,H,Jsmall,v₀small,Σ,iters_sm+1; method="adjfree",store_ens=true)

## Observation misfits
Θ = obsMisfits(V,H,ỹ)
Θsmall = obsMisfits(Vsmall,H,ỹsmall)

## State misfits
Ω = stateMisfits(V,H⁺,ỹ)
Ωsmall = stateMisfits(Vsmall,H⁺,ỹsmall)

## Spectral decompositions
𝒫,𝒬,𝒮,ℙ,ℚ,𝕊 = spectralproj(H,v₀,Σ)
𝒫sm,𝒬sm,𝒮sm,ℙsm,ℚsm,𝕊sm = spectralproj(H,v₀small,Σ)

## Compute norms
M = getLimitMisfits(ỹ,V,iters+1,H,Σ,Γ,Hₚ,𝒫,I-𝒫,ℙ,I-ℙ)
Msm = getLimitMisfits(ỹsmall,Vsmall,iters_sm+1,H,Σ,Γsmall,Hₚ,𝒫sm,I-𝒫sm,ℙsm,I-ℙsm)

## Plot misfits
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=5)]
X = range(0,iters,length = iters+1)
Xsm = range(0,iters_sm,length = iters_sm+1)
# Figure 1
fig = Figure(size=(900,750))

ax1 = Axis(fig[1,1];
    xlabel=L"Iterations $i$",
    ylabel=L"\text{Rel. }\text{\textbf{h}}_i^{(1:J)} \text{ error}",
    title=L"\text{Large ensemble}",
    yscale=log10,
    xlabelsize = 0,
    ylabelsize = 30,
    titlesize = 30,
    xticklabelsize = 0,
    yticklabelsize = 28,
)

ax2 = Axis(fig[2,1];
    xlabel=L"Iterations $i$",
    ylabel=L"\text{Rel. }\text{\textbf{v}}_i^{(1:J)} \text{ error }",
    title=L"\text{Small ensemble}",
    yscale=log10,
    xlabelsize = 28,
    ylabelsize = 30,
    titlesize = 0,
    xticklabelsize = 28,
    yticklabelsize = 28,
)

ax3 = Axis(fig[1,2];
    xlabel=L"Iterations $i$",
    title=L"\text{Small ensemble}",
    yscale=log10,
    ylabelvisible = false,
    yticksvisible = true,
    xlabelsize = 0,
    ylabelsize = 28,
    titlesize = 30,
    xticklabelsize = 0,
    yticklabelsize = 0,
)

ax4 = Axis(fig[2,2];
    xlabel=L"Iterations $i$",
    title=L"\text{Rel.}\text{\textbf{v}}_i^{(1:J)}\text{ error (small)}",
    yscale=log10,
    ylabelvisible = false,
    yticksvisible = true,
    xlabelsize = 28,
    ylabelsize = 28,
    titlesize = 0,
    xticklabelsize = 28,
    yticklabelsize = 0,
)

# Plot lines
lines!(ax1, X, vec(M.scrP.mean), label=L"\mathcal{P}\text{ space}", color=colors[2], linewidth=7)
lines!(ax1, X, vec(M.scrS.mean), label=L"\mathcal{S}\text{ space}", color=colors[4], linewidth=7, linestyle=:dash)
axislegend(ax1, position=:rt, labelsize=28, framevisible=false)

lines!(ax2, X, vec(M.bbP.mean), label=L"\text{\textbf{P}}\text{ space}", color=colors[2], linewidth=7, linestyle=:solid)
lines!(ax2, X, vec(M.bbS.mean), label=L"\text{\textbf{S}}\text{ space}", color=colors[4], linewidth=7, linestyle=:dash)
axislegend(ax2, position=:rt, labelsize=28, framevisible=false)

lines!(ax3, Xsm, vec(Msm.scrP.mean), label=L"\mathcal{P}\text{ space}", color=colors[2], linewidth=7, linestyle=:solid)
lines!(ax3, Xsm, vec(Msm.scrS.mean), label=L"\mathcal{S}\text{ space}", color=colors[4], linewidth=7, linestyle=:dash)
axislegend(ax3, position=:rt, labelsize=28, framevisible=false)

lines!(ax4, Xsm, vec(Msm.bbP.mean), label=L"\text{\textbf{P}}\text{ space}", color=colors[2], linewidth=7, linestyle=:solid)
lines!(ax4, Xsm, vec(Msm.bbS.mean), label=L"\text{\textbf{S}}\text{ space}", color=colors[4], linewidth=7, linestyle=:dash)
axislegend(ax4, position=:rt, labelsize=28, framevisible=false)

linkyaxes!(ax1, ax3)
linkyaxes!(ax2, ax4)

display(fig)
save("plots/misfit_means.pdf", fig)

## Fig 2
fig = Figure(size=(900,750))
ax1 = Axis(fig[1,1],
    xlabel=L"Iterations $i$",
    ylabel=L"\text{Rel. }\mathrm{Cov}[\text{\textbf{h}}_i^{(1:J)}] \text{ error}",
    title=L" \text{Large ensemble}",
    yscale=log10,
    xlabelsize = 0,
    ylabelsize = 30,
    titlesize = 30,
    xticklabelsize = 0,
    yticklabelsize = 28,
)
ax2 = Axis(fig[2,1],
    xlabel=L"Iterations $i$",
    ylabel=L"\text{Rel. }\mathrm{Cov}[\text{\textbf{v}}_i^{(1:J)}] \text{ error}",
    title=L"\text{ error (large)}",
    yscale=log10,
    xlabelsize = 28,
    ylabelsize = 30,
    titlesize = 0,
    xticklabelsize = 28,
    yticklabelsize = 28,
)
ax3 = Axis(fig[1,2],
    xlabel=L"Iterations $i$",
    title=L"\text{Small ensemble}",
    yscale=log10,
    xlabelsize = 0,
    ylabelsize = 0,
    titlesize = 30,
    xticklabelsize = 0,
    yticklabelsize = 0,
)
ax4 = Axis(fig[2,2],
    xlabel=L"Iterations $i$",
    title=L"\mathrm{Cov}[\text{\textbf{v}}_i^{(1:J)}] \text{ error (small)}",
    yscale=log10,
    xlabelsize = 28,
    ylabelsize = 0,
    titlesize = 0,
    xticklabelsize = 28,
    yticklabelsize = 0,
)

lines!(ax1, X, vec(M.scrP.cov), 
label = L"\mathcal{P}\text{ space}",
color = colors[2], linewidth = 7, linestyle=:solid)
lines!(ax1, X, vec(M.scrS.cov), label = L"\mathcal{S}\text{ space}",
color = colors[4], linewidth = 7, linestyle=:dash)
axislegend(ax1, position =:rt, labelsize = 28, framevisible=false)

lines!(ax2, X, vec(M.bbP.cov), label = L"\text{\textbf{P}}\text{ space}",
color = colors[2], linewidth = 7, linestyle=:solid)
lines!(ax2, X, vec(M.bbS.cov), label = L"\text{\textbf{S}}\text{ space}",
color = colors[4], linewidth = 7, linestyle=:dash)
axislegend(ax2, position =:rt, labelsize = 28, framevisible=false)
lines!(ax3, Xsm, vec(Msm.scrP.cov), label = L"\mathcal{P}\text{ space}",
color = colors[2], linewidth = 7, linestyle=:solid)
lines!(ax3, Xsm, vec(Msm.scrS.cov), label = L"\mathcal{S}\text{ space}",
color = colors[4], linewidth = 7, linestyle=:dash)
axislegend(ax3, position =:rt, labelsize = 28, framevisible=false)
lines!(ax4, Xsm, vec(Msm.bbP.cov), label = L"\text{\textbf{P}}\text{ space}",
color = colors[2], linewidth = 7, linestyle=:solid)
lines!(ax4, Xsm, vec(Msm.bbS.cov), label = L"\text{\textbf{S}}\text{ space}",
color = colors[4], linewidth = 7, linestyle=:dash)
axislegend(ax4, position =:rt, labelsize = 28, framevisible=false)
linkyaxes!(ax1, ax3)
linkyaxes!(ax2, ax4)
display(fig)
save(joinpath("plots", "misfit_covs.pdf"), fig)
