using EKRMLE
using LinearAlgebra
using CairoMakie
using PolynomialModelReductionDataset: Heat2DModel, integrate_model_fast, build_fast_be_solver, FastDenseSolver
using UniqueKronecker: invec
using Random
using Distributions
using MatrixEquations
using ColorSchemes
using SparseArrays
CairoMakie.activate!()

## Setup
heat = build_heat2d_params()
v0 = sample_prior_ic(heat)
C = two_block_observation_matrix(heat.d, 5, 5; sparse_output=false)
data = build_synthetic_data(heat, C, v0, 0.1; rel_noise_level=0.2)
# access pieces
U       = data.U
obs_idx = data.obs_idx
times   = data.obs_times
y_clean = data.y_clean
y       = data.y
Γ       = data.Γ

##
IC_field = reshape(v0, heat.Nx, heat.Ny)
fig1 = Figure()
ax1 = Axis3(fig1[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)")
sf = surface!(ax1, 0:heat.Δx:1, 0:heat.Δy:1, IC_field)
Colorbar(fig1[1, 2], sf)
fig1

##
final_field = reshape(U[:,end], heat.Nx, heat.Ny)
fig2 = Figure()
ax2 = Axis3(fig2[1, 1], xlabel="x", ylabel="y", zlabel="u(x,y,t)")
sf = surface!(ax2, 0:heat.Δx:1, 0:heat.Δy:1, final_field)
Colorbar(fig2[1, 2], sf)
fig2

##
# Explicit forward operator
H = build_explicit_forward_operator(heat, C, data.obs_idx)

# RLS data
yRLS, ΓRLS = build_RLS_data(data.y, data.Γ, heat.Γpr)

# Explicit HRLS wrapper
HRLS, HRLS_s = build_explicit_HRLS_operator(H)

# Implicit HRLS wrapper
H_of_v, HRLS_s_imp = build_implicit_HRLS_operator(heat, C, data.obs_idx) 
## Bayesian posterior
Γpos = (H'*(Γ\H) + heat.Γpr\I)\I
μpos = Γpos*H'*(Γ\y)

fig = Figure(size=(900,900))
ax1 = Axis(fig[1, 1], title = L"\textbf{Γ}_\text{pos}", titlesize=35)
ax1.yreversed=true
hm1 = heatmap!(ax1, Γpos; colormap=:magma)
Colorbar(fig[1, 2], hm1)
display(fig)

## EKRMLE
J = 1000
V0 = rand(MvNormal(vec(zeros(heat.d, 1)), Symmetric(heat.Γpr)), J)

## EKRMLE run (explicit)
#steps = 25
#ekrmleobj = EKRMLEObj(V0, yRLS, ΓRLS)
#EKRMLE_run!(ekrmleobj, nothing, HRLS_s, steps)

## EKRMLE run (implicit)
steps = 25
ekrmleobj = EKRMLEObj(V0, yRLS, ΓRLS)
@time EKRMLE_run!(ekrmleobj, nothing, HRLS_s_imp, steps)

##

@inline colmean(V::AbstractMatrix) = vec(mean(V, dims=2))

function _samplecov(V::AbstractMatrix)
    # computes sample covariance (column-wise)
    J = size(V, 2)
    μ = colmean(V)
    X = V .- μ
    return (X * X') / (J-1)
end

C_ens = _samplecov(ekrmleobj.V[end])
## Covariance comparison
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1], title=L"\text{Cov}[\textbf{v}_\text{end}^{(1:J)}]", titlesize=35)
ax1.yreversed=true
ax2 = Axis(fig[1, 3], title = L"\textbf{Γ}_\text{pos}", titlesize=35)
ax2.yreversed=true
hm1 = heatmap!(ax1, C_ens; colormap=:magma)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(ax2, Γpos; colormap=:magma)
Colorbar(fig[1, 4], hm2)
display(fig)
#save("plots/Heat_2D_fullcovcomp.png",fig)

##
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=5)]
μ = mean(ekrmleobj.V[end], dims=2)
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1], title = L"\text{Posterior mean comparison}", titlesize=35)
lines!(ax1, μpos; linewidth=7, label=L"𝛍_\text{pos}", color=colors[2])
lines!(ax1, vec(μ);linewidth=6, linestyle=:dash, label=L"\text{E}[\textbf{v}_\text{end}^{(1:J)}]", color=colors[4])
axislegend(ax1; position=:rb, framevisible = false, labelsize=35)
display(fig)
#save("plots/Heat_2D_fullmeancomp.svg",fig)

##
pos_field = reshape(μpos, heat.Nx, heat.Ny)
EKRMLE_field = reshape(μ, heat.Nx, heat.Ny)

fig = Figure(size=(1800,600))
ax1 = Axis(fig[1, 1], title=L"\text{E}[\textbf{v}_\text{end}^{(1:J)}]", titlesize=50)
ax1.yreversed=true
ax2 = Axis(fig[1, 3], title = L"𝛍_\text{pos}", titlesize=50)
ax2.yreversed=true
ax3 = Axis(fig[1, 5], title = L"\text{Truth}", titlesize=50)
ax3.yreversed=true
hm1 = heatmap!(ax1, EKRMLE_field; colormap=:magma)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(ax2, pos_field; colormap=:magma)
Colorbar(fig[1, 4], hm2)
hm3 = heatmap!(ax3, IC_field; colormap=:magma)
Colorbar(fig[1, 6], hm2)
display(fig)
#save("plots/Heat_2D_fullmeanfieldcomp.png",fig)

## Surface plots
Z1 = EKRMLE_field
Z2 = pos_field
Z3 = IC_field

nx, ny = size(Z1)
X1 = 1:nx
X2 = 1:ny

fig = Figure(size = (1800, 500))

ax1 = Axis3(fig[1, 1], title = L"\text{E}[\textbf{v}_\text{end}^{(1:J)}]", titlesize = 35)
ax2 = Axis3(fig[1, 3], title = L"𝛍_\text{pos}", titlesize = 35)
ax3 = Axis3(fig[1, 5], title = L"\text{Truth}", titlesize = 35)

s1 = surface!(ax1, X1, X2, Z1; colormap = :magma)
Colorbar(fig[1, 2], s1)

s2 = surface!(ax2, X1, X2, Z2; colormap = :magma)
Colorbar(fig[1, 4], s2)

s3 = surface!(ax3, X1, X2, Z3; colormap = :magma)
Colorbar(fig[1, 6], s3)

display(fig)
#save("plots/Heat_2D_fullmeansurfacecomp.png",fig)

## Errors
weighted_relative_error(μpos,vec(μ),Γpos)

## Model reduction
red = build_reduced_operators(heat, C, data.Γ; r=30)

Hhat = build_reduced_explicit_forward_operator(heat, red, data.obs_idx)

Hhat_of_v, HRLShat_s_imp, red_solver = build_reduced_HRLS_operator(
    heat, red, data.obs_idx
)

##
J = 10_000
V0 = rand(MvNormal(vec(zeros(heat.d, 1)), Symmetric(heat.Γpr)), J)
ekrmleobjBT = EKRMLEObj(V0, yRLS, ΓRLS)
steps = 25
@time EKRMLE_run!(ekrmleobjBT, nothing, HRLShat_s_imp, steps)

## Reuced posterior
ΓposBT = (Hhat'*(Γ\Hhat) + heat.Γpr\I)\I
μposBT = ΓposBT*Hhat'*(Γ\y)

fig = Figure(size=(900,900))
ax1 = Axis(fig[1, 1], title = L"\textbf{Γ}_\text{pos,BT}", titlesize=35)
ax1.yreversed=true
hm1 = heatmap!(ax1, Γpos; colormap=:magma)
Colorbar(fig[1, 2], hm1)
display(fig)

##
C_ensBT = _samplecov(ekrmleobjBT.V[end])
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1], title=L"\text{Cov}[\textbf{v}_\text{end,BT}^{(1:J)}]", titlesize=35)
ax1.yreversed=true
ax2 = Axis(fig[1, 3], title = L"\textbf{Γ}_\text{pos,BT}", titlesize=35)
ax2.yreversed=true
hm1 = heatmap!(ax1, C_ensBT; colormap=:magma)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(ax2, ΓposBT; colormap=:magma)
Colorbar(fig[1, 4], hm2)
display(fig)
#save("plots/Heat_2D_BTcovcomp_r5_J100k.png",fig)

##
colors = [get(ColorSchemes.magma, t) for t in range(0, stop=1, length=5)]
μBT = mean(ekrmleobjBT.V[end], dims=2)
fig = Figure(size=(900,400))
ax1 = Axis(fig[1, 1], title = L"\text{Posterior mean comparison}", titlesize=35)
lines!(ax1, μposBT; linewidth=7, label=L"𝛍_\text{pos,BT}", color=colors[2])
lines!(ax1, vec(μBT);linewidth=6, linestyle=:dash, label=L"\text{E}[\textbf{v}_\text{end,BT}^{(1:J)}]", color=colors[4])
axislegend(ax1; position=:rb, framevisible = false, labelsize=35)
display(fig)
#save("plots/Heat_2D_BTmeancomp_r5_J100k.png",fig)

##
pos_fieldBT = reshape(μposBT, heat.Nx, heat.Ny)
EKRMLE_fieldBT = reshape(μBT, heat.Nx, heat.Ny)

fig = Figure(size=(1800,600))
ax1 = Axis(fig[1, 1], title=L"\text{E}[\textbf{v}_\text{end,BT}^{(1:J)}]", titlesize=50)
ax1.yreversed=true
ax2 = Axis(fig[1, 3], title = L"𝛍_\text{pos,BT}", titlesize=50)
ax2.yreversed=true
ax3 = Axis(fig[1, 5], title = L"𝛍_\text{pos}", titlesize=50)
ax3.yreversed=true
hm1 = heatmap!(ax1, EKRMLE_fieldBT; colormap=:magma)
Colorbar(fig[1, 2], hm1)
hm2 = heatmap!(ax2, pos_fieldBT; colormap=:magma)
Colorbar(fig[1, 4], hm2)
hm3 = heatmap!(ax3, pos_field; colormap=:magma)
Colorbar(fig[1, 6], hm2)
display(fig)
#save("plots/Heat_2D_BTcmeanfieldcomp_r5_J100k.png",fig)

## Surface plots
Z1 = EKRMLE_fieldBT
Z2 = pos_fieldBT
Z3 = pos_field

nx, ny = size(Z1)
X1 = 1:nx
X2 = 1:ny

fig = Figure(size = (1800, 500))

ax1 = Axis3(fig[1, 1], title = L"\text{E}[\textbf{v}_\text{end,BT}^{(1:J)}]", titlesize = 35)
ax2 = Axis3(fig[1, 3], title = L"𝛍_\text{pos,BT}", titlesize = 35)
ax3 = Axis3(fig[1, 5], title = L"𝛍_\text{pos}", titlesize = 35)

s1 = surface!(ax1, X1, X2, Z1; colormap = :magma)
Colorbar(fig[1, 2], s1)

s2 = surface!(ax2, X1, X2, Z2; colormap = :magma)
Colorbar(fig[1, 4], s2)

s3 = surface!(ax3, X1, X2, Z3; colormap = :magma)
Colorbar(fig[1, 6], s3)

display(fig)
#save("plots/Heat_2D_BTmeansurfacecomp_r5_J100k.png",fig)

## Error of BT approximation
weighted_relative_error(μpos,vec(μposBT),Γpos)

## Error of EKRMLE
weighted_relative_error(μpos,vec(μBT),Γpos)