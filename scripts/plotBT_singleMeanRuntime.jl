using EKRMLE
using CairoMakie
using ColorSchemes
using JLD2
using Dates
using LaTeXStrings
CairoMakie.activate!()

## Load data
mc_results = load("data/mc_results_BT.jld2")

## Process data
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
