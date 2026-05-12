using EKRMLE
using CairoMakie
using ColorSchemes
using JLD2
using Dates
using LaTeXStrings
CairoMakie.activate!()

##
#@load "data/results_heat2d_bt_ekrmle_mc.jld2" mc_results
mc_results = load("data/mc_results_BT.jld2")
##
#=
Js = mc_results.Js
Rs = mc_results.Rs

full_mean_err = mc_results.full_mean_err_avg
full_cov_err  = mc_results.full_cov_err_avg
full_time     = mc_results.full_time_avg

red_mean_err = mc_results.red_mean_err_avg
red_cov_err  = mc_results.red_cov_err_avg
red_time     = mc_results.red_time_avg

bt_mean_err = mc_results.bt_mean_err_avg
bt_cov_err  = mc_results.bt_cov_err_avg

nJ = length(Js)
nR = length(Rs)

=#
##
Js = mc_results["Js"]
Rs = mc_results["Rs"]

full_mean_err = mc_results["full_mean_err"]
full_cov_err  = mc_results["full_cov_err"]
full_time     = mc_results["full_time"]

red_mean_err = mc_results["red_mean_err"]
red_cov_err  = mc_results["red_cov_err"]
red_time     = mc_results["red_time"]

bt_mean_err = mc_results["bt_mean_err"]
bt_cov_err  = mc_results["bt_cov_err"]

nJ = length(Js)
nR = length(Rs)

##
#=
jldsave("data/mc_results_BT.jld2";
    Js,
    Rs,
    full_mean_err,
    full_cov_err,
    full_time,
    red_mean_err,
    red_cov_err,
    red_time,
    bt_mean_err,
    bt_cov_err,
    nJ,
    nR,
)
=#
## 
colorsJ = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=nJ+2)]
colorsJ = colorsJ[end:-1:1]

fig = Figure(size=(1800, 600))

ax1 = Axis(fig[1,1],
    xlabel = L"\text{reduced model rank } r",
    title  = L"\mathrm{Average\ posterior\ mean\ error}",
    yscale = log10,
    xlabelsize = 38,
    ylabelsize = 38,
    titlesize = 40,
    xticklabelsize = 38,
    yticklabelsize = 38,
)

for j in 1:nJ
    J = Js[j]
    lines!(ax1, Rs, red_mean_err[j, :],
        linewidth = 10,
        color = colorsJ[j+1],
        label = latexstring("J = $(J)")
    )
end

lines!(ax1, Rs, bt_mean_err,
    linewidth = 10,
    color = :black,
    linestyle = :dash,
    label = L"\mathrm{BT}"
)

scatter!(ax1, Rs, bt_mean_err,
    color = :black,
    markersize = 30,
    marker = :circle,
)

ax2 = Axis(fig[1,2],
    xlabel = L"\text{reduced model rank } r",
    title  = L"\mathrm{Average\ posterior\ covariance\ error}",
    yscale = log10,
    xlabelsize = 38,
    ylabelsize = 0,
    titlesize = 40,
    xticklabelsize = 38,
    yticklabelsize = 0,
    ylabelvisible = false,
    yticksvisible = false,
)

for j in 1:nJ
    J = Js[j]
    lines!(ax2, Rs, red_cov_err[j, :],
        linewidth = 10,
        color = colorsJ[j+1],
        label = latexstring("J = $(J)")
    )
end

lines!(ax2, Rs, bt_cov_err,
    linewidth = 10,
    color = :black,
    linestyle = :dash,
    label = L"\mathrm{BT}"
)

scatter!(ax2, Rs, bt_cov_err,
    color = :black,
    markersize = 30,
    marker = :circle,
)

axislegend(ax2,
    position = :lb,
    labelsize = 38,
    merge = true,
    framevisible = false,
    patchsize = (70,40)
)

linkyaxes!(ax1, ax2)
display(fig)
# save("plots/avg_errors_vs_rank.pdf", fig)

##
colorsR = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=nR+1)]
colorsR = colorsR[end:-1:1]

fig1 = Figure(size=(1800, 600))

ax1 = Axis(fig1[1,1],
    xlabel = L"\text{ensemble size } J",
    title  = L"\text{Relative posterior mean error}",
    xscale = log10,
    yscale = log10,
    xlabelsize = 42,
    ylabelsize = 42,
    titlesize = 45,
    xticklabelsize = 38,
    yticklabelsize = 38,
)

for i in 2:nR-1
    r = Rs[i]
    lines!(ax1, Js, red_mean_err[:, i],
        linewidth = 12,
        color = colorsR[i],
        label = latexstring("r = $(r)")
    )
end

lines!(ax1, Js, full_mean_err,
    linewidth = 12,
    color = :black,
    linestyle = :dash,
    label = L"\mathrm{Full\ model}"
)

scatter!(ax1, Js, full_mean_err,
    color = :black,
    markersize = 30,
    marker = :circle,
)

axislegend(ax1,
    position = :lb,
    labelsize = 38,
    merge = true,
    framevisible = false,
    patchsize=(80,40)
)

ax2 = Axis(fig1[1,2],
    xlabel = L"\text{reduced model rank } r",
    title  = L"\text{Relative posterior mean error}",
    yscale = log10,
    xlabelsize = 42,
    ylabelsize = 0,
    titlesize = 45,
    xticklabelsize = 38,
    yticklabelsize = 0,
    ylabelvisible = false,
    yticksvisible = false,
)

for j in 1:nJ
    J = Js[j]
    lines!(ax2, Rs[1:end-1], red_mean_err[j, 1:end-1],
        linewidth = 12,
        color = colorsJ[j+1],
        label = latexstring("J = $(J)")
    )
end

lines!(ax2, Rs[1:end-1], bt_mean_err[1:end-1],
    linewidth = 12,
    color = :black,
    linestyle = :dash,
    label = L"\text{exact}"
)

scatter!(ax2, Rs[1:end-1], bt_mean_err[1:end-1],
    color = :black,
    markersize = 30,
    marker = :circle,
)

axislegend(ax2,
    position = :lb,
    labelsize = 38,
    merge = true,
    framevisible = false,
    patchsize = (30,30)
)

linkyaxes!(ax1, ax2)
display(fig1)
#save("plots/avg_mean_errors.pdf", fig1)
##
fig2 = Figure(size=(1800, 600))

ax1 = Axis(fig2[1,1],
    xlabel = L"\text{ensemble size } J",
    title  = L"\text{Relative posterior covariance error}",
    xscale = log10,
    yscale = log10,
    xlabelsize = 42,
    ylabelsize = 38,
    titlesize = 45,
    xticklabelsize = 38,
    yticklabelsize = 38,
)

for i in 2:nR-1
    r = Rs[i]
    lines!(ax1, Js, red_cov_err[:, i],
        linewidth = 12,
        color = colorsR[i],
        label = latexstring("r = $(r)")
    )
end

lines!(ax1, Js, full_cov_err,
    linewidth = 12,
    color = :black,
    linestyle = :dash,
    label = L"\mathrm{Full\ model}"
)

scatter!(ax1, Js, full_cov_err,
    color = :black,
    markersize = 30,
    marker = :circle,
)

axislegend(ax1,
    position = :lb,
    labelsize = 38,
    merge = true,
    framevisible = false,
    patchsize = (80,40)
)

ax2 = Axis(fig2[1,2],
    xlabel = L"\text{reduced model size } r",
    title  = L"\text{Relative posterior covariance error}",
    yscale = log10,
    xlabelsize = 42,
    ylabelsize = 0,
    titlesize = 45,
    xticklabelsize = 38,
    yticklabelsize = 0,
    ylabelvisible = false,
    yticksvisible = false,
)

for j in 1:nJ
    J = Js[j]
    lines!(ax2, Rs[1:end-1], red_cov_err[j, 1:end-1],
        linewidth = 12,
        color = colorsJ[j+1],
        label = latexstring("J = $(J)")
    )
end

lines!(ax2, Rs[1:end-1], bt_cov_err[1:end-1],
    linewidth = 12,
    color = :black,
    linestyle = :dash,
    label = L"\text{exact}"
)

scatter!(ax2, Rs[1:end-1], bt_cov_err[1:end-1],
    color = :black,
    markersize = 30,
    marker = :circle,
)

axislegend(ax2,
    position = :lb,
    labelsize = 38,
    merge = true,
    framevisible = false,
    patchsize = (80,40)
)

linkyaxes!(ax1, ax2)
display(fig2)
#save("plots/avg_cov_errors.pdf", fig2)

##
time_per_particle_full = full_time ./ Js
time_per_particle_red = similar(red_time)

for i in eachindex(Js), j in eachindex(Rs)
    time_per_particle_red[i, j] = red_time[i, j] / Js[i]
end

fig3 = Figure(size=(1800, 600))

ax1 = Axis(fig3[1,1],
    xlabel = L"\mathrm{average\ per\ particle\ cost}\; t/J",
    title  = L"\mathrm{Average\ mean\ error\ vs\ per\ particle\ cost}",
    xscale = log10,
    yscale = log10,
    xlabelsize = 38,
    ylabelsize = 38,
    titlesize = 40,
    xticklabelsize = 38,
    yticklabelsize = 38,
)

for j in 1:nR
    r = Rs[j]
    lines!(ax1, time_per_particle_red[:, j], red_mean_err[:, j],
        linewidth = 10,
        color = colorsR[j+1],
        label = latexstring("r = $(r)")
    )
    scatter!(ax1, time_per_particle_red[:, j], red_mean_err[:, j],
        color = colorsR[j+1],
        markersize = 24
    )
end

lines!(ax1, time_per_particle_full, full_mean_err,
    linewidth = 10,
    color = :black,
    linestyle = :dash,
    label = L"\mathrm{Full\ model}"
)
scatter!(ax1, time_per_particle_full, full_mean_err,
    color = :black,
    markersize = 30,
    marker = :circle
)

axislegend(ax1,
    position = :rt,
    labelsize = 30,
    merge = true,
    framevisible = false
)

ax2 = Axis(fig3[1,2],
    xlabel = L"\mathrm{average\ per\ particle\ cost }\; t/J",
    title  = L"\mathrm{Average\ covariance\ error\ vs\ per\ particle\ cost}",
    xscale = log10,
    yscale = log10,
    xlabelsize = 38,
    ylabelsize = 0,
    titlesize = 40,
    xticklabelsize = 38,
    yticklabelsize = 0,
    ylabelvisible = false,
    yticksvisible = false,
)

for j in 1:nR
    r = Rs[j]
    lines!(ax2, time_per_particle_red[:, j], red_cov_err[:, j],
        linewidth = 10,
        color = colorsR[j+1],
        label = latexstring("r = $(r)")
    )
    scatter!(ax2, time_per_particle_red[:, j], red_cov_err[:, j],
        color = colorsR[j+1],
        markersize = 24
    )
end

lines!(ax2, time_per_particle_full, full_cov_err,
    linewidth = 10,
    color = :black,
    linestyle = :dash,
    label = L"\mathrm{Full\ model}"
)
scatter!(ax2, time_per_particle_full, full_cov_err,
    color = :black,
    markersize = 30,
    marker = :circle
)

axislegend(ax2,
    position = :rt,
    labelsize = 30,
    merge = true,
    framevisible = false
)

linkyaxes!(ax1, ax2)
display(fig3)
# save("plots/avg_error_vs_per_particle_cost.pdf", fig3)

## time per particle
time_per_particle_full = full_time ./ Js
time_per_particle_red = similar(red_time)

for i in eachindex(Js), j in eachindex(Rs)
    time_per_particle_red[i, j] = red_time[i, j] / Js[i]
end

# Colors by reduced rank
nR = length(Rs)
colorsR = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=nR+4)]
colorsR = colorsR[end:-1:1]

fig = Figure(size=(1800, 600))

# --------------------------------------------------
# Mean error vs per-particle cost
# --------------------------------------------------
ax1 = Axis(fig[1,1],
    xlabel = L"\text{per particle cost } t/J",
    title  = L"\text{Relative posterior mean error}",
    xscale = log10,
    yscale = log10,
    xlabelsize = 42,
    ylabelsize = 38,
    titlesize = 45,
    xticklabelsize = 38,
    yticklabelsize = 38,
)

for j in 1:nR
    r = Rs[j]
    lines!(ax1, time_per_particle_red[:, j], red_mean_err[:, j],
        linewidth = 10,
        color = colorsR[j+1],
        label = latexstring("r = $(r)")
    )
    scatter!(ax1, time_per_particle_red[:, j], red_mean_err[:, j],
        color = colorsR[j+1],
        markersize = 24
    )
end

lines!(ax1, time_per_particle_full, full_mean_err,
    linewidth = 10,
    color = :black,
    linestyle = :dash,
    label = L"\mathrm{Full\ model}"
)
scatter!(ax1, time_per_particle_full, full_mean_err,
    color = :black,
    markersize = 30,
    marker = :circle
)
#=
axislegend(ax1,
    position = :rt,
    labelsize = 30,
    merge = true,
    framevisible = false
)
=#

# --------------------------------------------------
# Covariance error vs per-particle cost
# --------------------------------------------------
ax2 = Axis(fig[1,2],
    xlabel = L"\text{per particle cost } t/J",
    title  = L"\text{Relative posterior covariance error}",
    xscale = log10,
    yscale = log10,
    xlabelsize = 42,
    ylabelsize = 0,
    titlesize = 45,
    xticklabelsize = 38,
    yticklabelsize = 0,
    ylabelvisible = false,
    yticksvisible = false,
)

for j in 1:nR
    r = Rs[j]
    lines!(ax2, time_per_particle_red[:, j], red_cov_err[:, j],
        linewidth = 10,
        color = colorsR[j+1],
        label = latexstring("r = $(r)")
    )
    scatter!(ax2, time_per_particle_red[:, j], red_cov_err[:, j],
        color = colorsR[j+1],
        markersize = 24
    )
end

lines!(ax2, time_per_particle_full, full_cov_err,
    linewidth = 10,
    color = :black,
    linestyle = :dash,
    label = L"\mathrm{Full\ model}"
)
scatter!(ax2, time_per_particle_full, full_cov_err,
    color = :black,
    markersize = 30,
    marker = :circle
)

axislegend(ax2,
    position = :cb,
    labelsize = 38,
    merge = true,
    framevisible = false,
    patchsize = (70, 40)
)

linkyaxes!(ax1, ax2)

display(fig)
save("plots/error_vs_per_particle_cost.pdf", fig)

##

# Colors by reduced rank
colorsR = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=nR+1)]
colorsR = colorsR[end:-1:1]

fig = Figure(size=(1800, 600))

# --------------------------------------------------
# Mean error vs full runtime
# --------------------------------------------------
ax1 = Axis(fig[1,1],
    xlabel = L"\text{wallclock time (seconds)}",
    title  = L"\text{Relative posterior mean error}",
    xscale = log10,
    yscale = log10,
    xlabelsize = 42,
    ylabelsize = 38,
    titlesize = 45,
    xticklabelsize = 38,
    yticklabelsize = 38,
)

for j in 2:nR-1
    r = Rs[j]
    lines!(ax1, red_time[:, j], red_mean_err[:, j],
        linewidth = 12,
        color = colorsR[j],
        label = latexstring("r = $(r)")
    )
    scatter!(ax1, red_time[:, j], red_mean_err[:, j],
        color = colorsR[j],
        markersize = 30
    )
end

lines!(ax1, full_time, full_mean_err,
    linewidth = 12,
    color = :black,
    linestyle = :dash,
    label = L"\mathrm{Full\ model}"
)
scatter!(ax1, full_time, full_mean_err,
    color = :black,
    markersize = 30,
    marker = :circle
)

axislegend(ax1,
    position = :rt,
    labelsize = 38,
    merge = true,
    framevisible = false,
    patchsize = (80, 40)
)


# --------------------------------------------------
# Covariance error vs full runtime
# --------------------------------------------------
ax2 = Axis(fig[1,2],
    xlabel = L"\text{wallclock time (seconds)}",
    title  = L"\text{Relative posterior covariance error}",
    xscale = log10,
    yscale = log10,
    xlabelsize = 42,
    ylabelsize = 0,
    titlesize = 45,
    xticklabelsize = 38,
    yticklabelsize = 0,
    ylabelvisible = false,
    yticksvisible = false,
)

for j in 2:nR-1
    r = Rs[j]
    lines!(ax2, red_time[:, j], red_cov_err[:, j],
        linewidth = 12,
        color = colorsR[j],
        label = latexstring("r = $(r)")
    )
    scatter!(ax2, red_time[:, j], red_cov_err[:, j],
        color = colorsR[j],
        markersize = 30
    )
end

lines!(ax2, full_time, full_cov_err,
    linewidth = 12,
    color = :black,
    linestyle = :dash,
    label = L"\mathrm{Full\ model}"
)
scatter!(ax2, full_time, full_cov_err,
    color = :black,
    markersize = 30,
    marker = :circle
)
#=
axislegend(ax2,
    position = :rt,
    labelsize = 38,
    merge = true,
    framevisible = false,
    patchsize = (70, 40)
)
=#

linkyaxes!(ax1, ax2)

display(fig)
#save("plots/avg_error_vs_runtime.pdf", fig)

## Plot left panel of runtime plot only
fig = Figure(size=(900, 600))

# --------------------------------------------------
# Mean error vs full runtime
# --------------------------------------------------
ax1 = Axis(fig[1,1],
    xlabel = L"\text{wallclock time (seconds)}",
    title  = L"\text{Relative posterior mean error}",
    xscale = log10,
    yscale = log10,
    xlabelsize = 42,
    ylabelsize = 38,
    titlesize = 45,
    xticklabelsize = 38,
    yticklabelsize = 38,
)
j = 3 # index for r = 100 line
r = Rs[j]

r = Rs[j]
lines!(ax1, red_time[:, j], red_mean_err[:, j],
    linewidth = 12,
    color = colorsR[j],
    label = latexstring("r = $(r)")
)
scatter!(ax1, red_time[:, j], red_mean_err[:, j],
    color = colorsR[j],
    markersize = 30
)

lines!(ax1, full_time, full_mean_err,
    linewidth = 12,
    color = :black,
    linestyle = :dash,
    label = L"\mathrm{Full\ model}"
)
scatter!(ax1, full_time, full_mean_err,
    color = :black,
    markersize = 30,
    marker = :circle
)

axislegend(ax1,
    position = :rt,
    labelsize = 38,
    merge = true,
    framevisible = false,
    patchsize = (80, 40)
)



display(fig)
#save("plots/mean_error_vs_runtime.pdf", fig)
