export Darcy_2D_solver, get_logk_2D, Darcy_params_2D, get_data, KL_expansion_2D, fwd_2D, fwd_RLS_2D, plot_field, plot_field_sbs

# code was adapted, with permission from the author, from: https://github.com/PKU-CMEGroup/InverseProblems.jl/blob/master/Fluid/Darcy-2D.jl


mutable struct Darcy_params_2D{T<:AbstractFloat, TI<:Int}
    # Discretization
    Nₓ::TI # mesh size
    L::T # domain
    Δx::T # discretization level
    X::Vector{T} # mesh

    # source term
    f::Matrix{T}

    # Covariance parameters
    α::T
    τ::T

    # KL parameters
    N_KL::TI
    logk_2d::Matrix{T}
    φ::Array{T, 3}
    λ::Vector{T}
    θ_true::Vector{T}

    # inversion
    y_locs::Vector{TI}
    x_locs::Vector{TI}
    n::TI
    d::TI

end

function Darcy_params_2D(
    Nₓ::TI,
    L::T,
    N_KL::TI,
    Δ_obs_X::TI,
    Δ_obs_Y::TI,
    d::TI,
    α::T,
    τ::T,
    σ₀::T=1.0

) where {T<:AbstractFloat,TI<:Int}

    X = Vector(LinRange(0,L,Nₓ))
    y_locs = Vector(Δ_obs_Y:Δ_obs_Y:Nₓ-Δ_obs_Y)
    x_locs = Vector(Δ_obs_X:Δ_obs_X:Nₓ-Δ_obs_X)
    n = length(y_locs)
    Δx = X[2]-X[1]
    f = eff_2D(X)
    logK,φ,λ,θ_true = KL_expansion_2D(X,N_KL,α,τ,σ₀)

    Darcy_params_2D(Nₓ,L,Δx,X,f,α,τ,N_KL,logK,φ,λ,θ_true,y_locs,x_locs,n,d)
    
end

function eff_2D(y::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(y)
    f = zeros(T, N, N)
    for i = 1:N
        if y[i] <= 2/6
            f[:, i] .= 1000.0
        elseif (y[i] > 2/6 && y[i] <= 4/6)
            f[:, i] .= 3000.0
        elseif (y[i] > 4/6 && y[i] <= 5/6)
            f[:, i] .= 5000.0
        elseif y[i] > 5/6
            f[:, i] .= 9000.0
        end

    end
    return f
    
end



function get_pairs(N_KL::TI) where {TI<:Int}
    m = floor(Int, sqrt(2*N_KL)) + 1
    N_pairs = (m + 1)^2 - 1

    pairs = Array{Int}(undef, N_pairs, 2)
    mags = Array{Int}(undef, N_pairs)

    k = 0
    for i = 0:m, j = 0:m
        if i == 0 && j == 0
            continue
        end
        k += 1
        pairs[k, 1] = i
        pairs[k, 2] = j
        mags[k] = i^2 + j^2

    end

    pairs = pairs[sortperm(mags), :]
    return pairs[1:N_KL, :]
    
end


function KL_expansion_2D(x::Array{flt,1}, Nₖ::int, α::flt=2.0, τ::flt=3.0, σ₀::flt=1.0; rng::AbstractRNG=Random.default_rng()) where {flt<:AbstractFloat, int<:Int}
    N = length(x)
    X, Y = repeat(x, 1, N), repeat(x, 1, N)'
    pairs = get_pairs(Nₖ)

    φ = zeros(flt, N, N, Nₖ)
    λ = zeros(flt, Nₖ)

    for i = 1:Nₖ
        if (pairs[i, 1] == 0 && pairs[i, 2] == 0)
            φ[:, :, i] .= 1
        elseif pairs[i, 1] == 0
            φ[:, :, i] = sqrt(2)*cos.(pi * (pairs[i, 2]*Y))
        elseif pairs[i, 2] == 0
            φ[:, :, i] = sqrt(2)*cos.(pi * (pairs[i, 1]*X))
        else
            φ[:, :, i] = 2*cos.(pi * (pairs[i, 2]*Y)) .* cos.(pi * (pairs[i, 1]*X))
        end
        
        λ[i] = (pi^2 * (pairs[i, 1]^2 + pairs[i, 2]^2) + τ^2)^(-α)

    end

    𝛉 = rand(rng, Normal(0, σ₀), Nₖ)

    logk_2D = zeros(flt, N, N)
    for i = 1:Nₖ
        logk_2D .+= 𝛉[i]*sqrt(λ[i])*φ[:, :, i]
    end

    return logk_2D, φ, λ, 𝛉

    
end


function ind(darcy::Darcy_params_2D{T,TI}, xi::TI, yi::TI) where {T<:AbstractFloat,TI<:Int}
    return (xi-1) + (yi-2)*(darcy.Nₓ - 2)
    
end

function Darcy_2D_solver(darcy::Darcy_params_2D{T,TI}, k::Array{T,2}) where {T<:AbstractFloat,TI<:Int}
    Δx, N = darcy.Δx, darcy.Nₓ
    C = 2*(Δx^2)
    f = darcy.f

    indX = TI[]
    indY = TI[]
    vals = T[]

    for yi = 2:N-1
        for xi = 2:N-1
            xyi = ind(darcy, xi, yi)

            # top
            if yi == N-1
                push!(indX, xyi)
                push!(indY, xyi)
                push!(vals, (k[xi,yi]+k[xi,yi+1])/C)
            else
                append!(indX, [xyi, xyi])
                append!(indY, [xyi, ind(darcy, xi, yi+1)])
                append!(vals, [(k[xi, yi] + k[xi, yi+1])/C, -(k[xi, yi] + k[xi, yi+1])/C])
            end

            # bottom
            if yi == 2
                push!(indX, xyi)
                push!(indY, xyi)
                push!(vals, (k[xi,yi]+k[xi,yi-1])/C)
            else
                append!(indX, [xyi, xyi])
                append!(indY, [xyi, ind(darcy, xi, yi-1)])
                append!(vals, [(k[xi, yi] + k[xi, yi-1])/C, -(k[xi, yi] + k[xi, yi-1])/C])
            end

            # right
            if xi == N-1
                push!(indX, xyi)
                push!(indY, xyi)
                push!(vals, (k[xi,yi]+k[xi+1,yi])/C)
            else
                append!(indX, [xyi, xyi])
                append!(indY, [xyi, ind(darcy, xi+1, yi)])
                append!(vals, [(k[xi, yi] + k[xi+1, yi])/C, -(k[xi, yi] + k[xi+1, yi])/C])
            end

            # left
            if xi == 2
                push!(indX, xyi)
                push!(indY, xyi)
                push!(vals, (k[xi,yi]+k[xi-1,yi])/C)
            else
                append!(indX, [xyi, xyi])
                append!(indY, [xyi, ind(darcy, xi-1, yi)])
                append!(vals, [(k[xi, yi] + k[xi-1, yi])/C, -(k[xi, yi] + k[xi-1, yi])/C])
            end
            
        end
    end


    df = sparse(indX, indY, vals, (N-2)^2, (N-2)^2)
    h = df\(f[2:N-1, 2:N-1])[:]

    h2d = zeros(T, N, N)
    h2d[2:N-1, 2:N-1] .= reshape(h, N-2, N-2)

    return h2d

end


function get_data(darcy::Darcy_params_2D{T,TI}, h::Array{T,2}) where {T<:AbstractFloat,TI<:Int}
    data = h[darcy.x_locs, darcy.y_locs]

    return data[:] 
end


function get_logk_2D(darcy::Darcy_params_2D{T,TI}, v::AbstractVector{T}) where {T<:AbstractFloat, TI<:Int}
    N, N_KL = darcy.Nₓ, darcy.N_KL
    λ, φ = darcy.λ, darcy.φ
    d = darcy.d

    @assert(d <= N_KL)
    logk = zeros(T, N, N)
    for i = 1 : d
        logk .+= v[i] * sqrt(λ[i]) * φ[:, :, i]
    end

    return logk
    
end

function fwd_2D(darcy::Darcy_params_2D{T,TI}, v::AbstractVector{T}) where {T<:AbstractFloat, TI<:Int}

    logk = get_logk_2D(darcy, v)
    k = exp.(logk)
    h = Darcy_2D_solver(darcy, k)
    y = get_data(darcy, h)

    return y
    
end


function fwd_RLS_2D(darcy::Darcy_params_2D{T,TI}, v::AbstractVector{T}) where {T<:AbstractFloat, TI<:Int}
    y = fwd_2D(darcy, v)
    return [y ; v]
end


function plot_field(darcy::Darcy_params_2D{T, TI}, u::Array{T, 2}, obs::Bool=false, filename::String="None") where {T<:AbstractFloat,TI<:Int}
    N = darcy.Nₓ
    X = darcy.X

    fig = Figure(size = (600,600))
    ax = Axis(fig[1,1], title = "log field")

    XX, YY = repeat(X, 1, N), repeat(X, 1, N)'
    hm = heatmap!(ax, X, X, u, colormap=:magma)
    Colorbar(fig[1,2], hm)

    if obs
        x_obs, y_obs = X[darcy.x_locs], X[darcy.y_locs]
        x_pts = repeat(x_obs, inner = length(y_obs))
        y_pts = repeat(y_obs, outer = length(x_obs))
        scatter!(ax, x_pts, y_pts; color=:black, markersize = 15)
    end

    return (fig, ax)

end


function plot_field!(ax::Axis, darcy::Darcy_params_2D{T,TI}, u::AbstractMatrix{T}; colormap=:magma, colorrange=nothing) where {T<:AbstractFloat,TI<:Int}
    X = darcy.X
    hm = heatmap!(ax, X, X, u; colormap=colormap, colorrange=colorrange)
    return hm
end

function plot_field_sbs(darcy::Darcy_params_2D{T,TI}, truth::AbstractMatrix{T}, approx::AbstractMatrix{T};
    titles::Tuple{String,String}=("Truth", "EKRMLE"), colormap=:magma) where {T<:AbstractFloat,TI<:Int}

    #lo = min(minimum(truth), minimum(approx))
    #hi = max(maximum(truth), maximum(approx))
    #clim = (lo, hi)

    lo_truth = minimum(truth)
    hi_truth = maximum(truth)
    clim_truth = (lo_truth, hi_truth)

    lo_aprx = minimum(approx)
    hi_aprx = maximum(approx)
    clim_aprx = (lo_aprx, hi_aprx)


    fig = Figure(size=(1200,600))
    ax1 = Axis(fig[1, 1], title=titles[1])
    ax2 = Axis(fig[1, 3], title=titles[2])

    #hm1 = plot_field!(ax1, darcy, truth; colormap=colormap, colorrange=clim)
    #hm2 = plot_field!(ax2, darcy, approx; colormap=colormap, colorrange=clim)

    hm1 = plot_field!(ax1, darcy, truth; colormap=colormap, colorrange=clim_truth)
    hm2 = plot_field!(ax2, darcy, approx; colormap=colormap, colorrange=clim_aprx)
    
    Colorbar(fig[:, 2], hm1)
    Colorbar(fig[:, 4], hm2)
    display(fig)
end




