using EKRMLE
using CairoMakie
using ControlSystems
using LinearAlgebra
using Distributions
using ProgressMeter
using LaTeXStrings
using MAT
using ColorSchemes
## Setup
# Matrices as in "heat-cont.mat"
A, C = getData()
d = size(A,2) # state dimension
n = 100 # number of observations
B = I(d)
σ_obs = 0.008 # noise std
Δt = 0.001 # time step
t_stop = 10
h = t_stop/n # time between measurements
t_bar = h:h:t_stop # measurement times
T = 0.0:Δt:t_stop # time domain

μ_pr = zeros(d) # prior mean
Γ_pr = getGamPr(A,B)
X₀ = generate_ensemble(1,μ_pr,Γ_pr) # initial condition

# Solve using Forward Euler
Y = solveHE(A,C,X₀,Δt,T)

# Build forward operator G
G = getG(A,C,h,Δt,n)

# Construct noise covariance
γₑ = fill(σ_obs^2,n)
Γ_obs = Diagonal(γₑ)

# Get noisy measurements
m = getNoisy(Y,Γ_obs,h,Δt)

# Get operator F
z,F,Σ = get_regularized(m,Γ_pr,Γ_obs,G,μ_pr)

# Plot solution & measurements
fig = Figure()
ax = Axis(fig[1,1],
    xlabel=L"Index",
    #ylabel=L"λ_i",
    #yscale=log10
)

scatter!(ax, T, vec(Y), label= L"Y")
scatter!(ax, t_bar, vec(m), marker=:cross, label = L"m")
axislegend(position =:lb, labelsize = 20)
display(fig)

## True Posterior
Fish = (G'/Γ_obs)*G
Γₚ = (Fish + Γ_pr\I)\I
μₚ = (Γₚ*G'/Γ_obs)*m

## Model reduction setup
MC_runs = 30
Js = [1000,10000,100000,1000000]
Iter = [200,100,30,30]
Rs = [3,5,10,20]
rs = size(Rs,1)
js = size(Js,1)
μ_BT_errs = zeros(js,rs,MC_runs)
Γ_BT_errs = similar(μ_BT_errs)
μ_BTEKI_errs = similar(μ_BT_errs)
Γ_BTEKI_errs = similar(μ_BT_errs)
μ_EKI_errs = zeros(js,1,MC_runs)
Γ_EKI_errs = similar(μ_EKI_errs)
C = Matrix(C)
A = Matrix(A)
C = Float64.(C)
R = cholesky(Γ_pr).L' # get R factor
L = matread("data/L_factor.mat")["L"] # L factor obtained via lyapchol()
UU,S,VV = svd(L'R)

## MC runs
@showprogress for k = 1 : MC_runs
    V₀ = generate_ensemble(maximum(Js),μ_pr,Γ_pr)
    ## full EKI
    outer = Progress(js, desc="full EKRMLE runs")
    for j = 1 : js
        J = Js[j]
        v₀ = V₀[:,1:J]
        iters = Iter[j]
        # EKI algorithm
        V_full,Γ_full,ỹ_full = EKR(z,F,J,v₀,Σ,iters+1; method="adjfree",store_ens=false)
        μ_EKI_full = V_full[:,end]
        # full EKI errors
        μ_EKI_errs[j,1,k] = (sqrt(((μₚ-μ_EKI_full)'/Γₚ)*(μₚ-μ_EKI_full))/sqrt((μₚ'/Γₚ)*μₚ))[1]
        Γ_EKI_errs[j,1,k] = opnorm(Γₚ-Γ_full)/opnorm(Γₚ)
        next!(outer)
    end

    ## reduced models
    for j = 1 : rs
        # reduce system using BT for a given r 
        r = Rs[j]
        Uᵣ = UU[:,1:r]
        Vᵣ = VV[:,1:r]
        Sᵣ = S[1:r]
        Σᵣ = Diagonal(1 ./ sqrt.(Sᵣ))
        # build bases
        Wᵣ_tild_tr = Σᵣ*Uᵣ'L' 
        Wᵣ = R*Vᵣ*Σᵣ
        # build reduced operators
        Â = Wᵣ_tild_tr*A*Wᵣ
        Ĉ = C*Wᵣ
        # simulate using reduced operators
        Ĝ = getG(Â,Ĉ,h,Δt,n)
        Ĝ = Ĝ*Wᵣ_tild_tr
        ~,F̂,~ = get_regularized(m,Γ_pr,Γ_obs,Ĝ,μ_pr)
        # reduced posterior
        Fish_BT = (Ĝ'/Γ_obs)*Ĝ
        Γ_pos_BT = (Fish_BT + Γ_pr\I)\I
        μ_pos_BT = (Γ_pos_BT*Ĝ'/Γ_obs)*m
        inner = Progress(js, desc="EKRMLE runs for r=$r")
        for i = 1 : js
            # EKRMLE setup
            J = Js[i]
            v₀ = V₀[:,1:J]
            iters = Iter[i]
            # BTEKRMLE algorithm
            V,Γ,ỹ = EKR(z,F̂,J,v₀,Σ,iters+1; method="adjfree",store_ens=false)
            μ_EKI = V[:,end]
            μ_BT_errs[i,j,k] = (sqrt(((μₚ-μ_pos_BT)'/Γₚ)*(μₚ-μ_pos_BT))/sqrt((μₚ'/Γₚ)*μₚ))[1]
            Γ_BT_errs[i,j,k] = opnorm(Γₚ-Γ_pos_BT)/opnorm(Γₚ)
            # BTEKRMLE errors
            μ_BTEKI_errs[i,j,k] = (sqrt(((μₚ-μ_EKI)'/Γₚ)*(μₚ-μ_EKI))/sqrt((μₚ'/Γₚ)*μₚ))[1]
            Γ_BTEKI_errs[i,j,k] = opnorm(Γₚ-Γ)/opnorm(Γₚ)
            next!(inner)
        end
    end
end



## Plot full runs
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=rs+2)]
colors = colors[end:-1:1]
fig = Figure(size=(750,500))
ax1 = Axis(fig[1,1],
    xlabel=L"ensemble size $J$",
    title=L"Error in $μ_p$",
    #ylabel=L"λ_i",
    yscale=log10,
    xscale=log10
)
#X = range(0,iters,length = iters+1)
for j = 1 : rs
    r = Rs[j]
    lines!(ax1, Js, vec(mean(μ_BTEKI_errs,dims=3)[:,j]),
     linewidth=4,
     color = colors[j+1],
     label = latexstring("r = $(r)"))
end
#scatterlines!(ax1, Rs, vec(μ_BTEKI_errs[4,:]), label = L"J = 100")
lines!(ax1, Js, vec(mean(μ_EKI_errs,dims=3)),
    linewidth = 3, color = :black, linestyle=:dash,
    label = "Full EKI"
)
scatter!(
    ax1, Js, vec(mean(μ_EKI_errs,dims=3)),
    color=:black, markersize=15,
    marker =:circle,
    label = "Full EKI"
)
axislegend(ax1,position =:lb, labelsize = 25,
merge=true,framevisible=false)
display(fig)

fig = Figure(size=(750,500))
ax1 = Axis(fig[1,1],
    xlabel=L"ensemble size $J$",
    title=L"Error in $Γ_p$",
    #ylabel=L"λ_i",
    yscale=log10,
    xscale=log10
)
#X = range(0,iters,length = iters+1)
for j = 1 : rs
    r = Rs[j]
    lines!(ax1, Js, vec(mean(Γ_BTEKI_errs,dims=3)[:,j]),
     linewidth=4,
     color = colors[j+1],
     label = latexstring("r = $(r)"))
end
#scatterlines!(ax1, Rs, vec(μ_BTEKI_errs[4,:]), label = L"J = 100")
lines!(ax1, Js, vec(mean(Γ_EKI_errs,dims=3)),
    linewidth = 3, color = :black, linestyle=:dash,
    label = "Full EKI"
)
scatter!(
    ax1, Js, vec(mean(Γ_EKI_errs,dims=3)),
    color=:black, markersize=15,
    marker =:circle,
    label = "Full EKI"
)
axislegend(ax1,position =:lb, labelsize = 25,
merge=true,framevisible=false)
display(fig)

## Plot BT EKI runs
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=js+2)]
colors = colors[end:-1:1]
fig = Figure(size=(750,500))
ax1 = Axis(fig[1,1],
    xlabel=L"reduced model size $r$",
    title=L"Error in $μ_p$",
    #ylabel=L"λ_i",
    yscale=log10,
)
for j = 1 : js
    J = Js[j]
    lines!(ax1, Rs, vec(mean(μ_BTEKI_errs,dims=3)[j,:]),
     linewidth=4,
     color = colors[j+1],
     label = latexstring("J = $(J)"))
end
#scatterlines!(ax1, Rs, vec(μ_BTEKI_errs[4,:]), label = L"J = 100")
lines!(ax1, Rs, vec(mean(μ_BT_errs,dims=3)[1,:]),
    linewidth = 3, color = :black, linestyle=:dash,
    label = L"J = \infty"
)
scatter!(
    ax1, Rs, vec(mean(μ_BT_errs,dims=3)[1,:]),
    color=:black, markersize=15,
    marker =:circle,
    label = L"J = \infty"
)
axislegend(ax1,position =:lb, labelsize = 25,
merge=true,framevisible=false)
display(fig)

fig = Figure(size=(750,500))
ax1 = Axis(fig[1,1],
    xlabel=L"reduced model size $r$",
    title=L"Error in $Γ_p$",
    #ylabel=L"λ_i",
    yscale=log10,
)
for j = 1 : js
    J = Js[j]
    lines!(ax1, Rs, vec(mean(Γ_BTEKI_errs,dims=3)[j,:]),
     linewidth=4,
     color = colors[j+1],
     label = latexstring("J = $(J)"))
end
#scatterlines!(ax1, Rs, vec(μ_BTEKI_errs[4,:]), label = L"J = 100")
lines!(ax1, Rs, vec(mean(Γ_BT_errs,dims=3)[1,:]),
    linewidth = 3, color = :black, linestyle=:dash,
    label = L"J = \infty"
)
scatter!(
    ax1, Rs, vec(mean(Γ_BT_errs,dims=3)[1,:]),
    color=:black, markersize=15,
    marker =:circle,
    label = L"J = \infty"
)
axislegend(ax1,position =:lb, labelsize = 25,
merge=true,framevisible=false)
display(fig)

## Save data
#=
matwrite(
    "data/BTEKI.mat", Dict(
        "mu_BTEKI_errs" => μ_BTEKI_errs,
        "mu_EKI_errs" => μ_EKI_errs,
        "mu_BT_errs" => μ_BT_errs,
        "Gamma_BT_errs" => Γ_BT_errs,
        "Gamma_EKI_errs" => Γ_EKI_errs,
        "Gamma_BTEKI_errs" => Γ_BTEKI_errs,
        "MC_runs" => MC_runs,
        "Js" => Js,
        "Iters" => Iter,
        "Rs" => Rs
    )
)
=#