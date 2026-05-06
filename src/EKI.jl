export EKIObj, EKI_run!, EKI_run!


mutable struct EKIObj{T<:AbstractFloat, TI<:Int}
    V::Vector{Matrix{T}}
    y::Vector{T}
    Γ::Matrix{T}
    Σ::Matrix{T}
    Ohist::Vector{Matrix{T}}
    J::TI; d::TI; n::TI
    iters::TI
end

function EKIObj(
    V₀::AbstractMatrix{T}, # (d × J) columns are v₀^{(j)}
    y::AbstractVector{T}, # length n
    Γ::AbstractMatrix{T}, # (n × n)
    Σ::AbstractMatrix{T};
    rng::AbstractRNG=Random.default_rng(),
) where {T<:AbstractFloat}
    d, J = size(V₀)
    n = length(y)
    EKIObj{T,Int}(
        [Matrix{T}(V₀)],
        Vector{T}(y),
        Matrix{T}(Γ),
        Matrix{T}(Σ),
        Matrix{T}[],
        J, d, n,
        0
    )
end

function EKI_step!(obj::EKIObj{T}, Hens::Function, flavor::String="vanilla") where {T<:AbstractFloat}
    # Current ensemble and forward evaluation
    V = obj.V[end]
    O = Hens(V)
    
    # Covariances
    Chh = _samplecov(O)
    Cvh = _samplecrosscov(V, O)

    # Compute Kalman gain
    A = Chh + obj.Γ
    AL = cholesky(A; check=false)
    n = size(O, 1)
    #K = Cvh*(AL \ I(n))
    K = Cvh/A

    # Residuals
    Θ = obj.y .- O

    # Update ensemble
    if flavor == "vanilla"
        Vnew = similar(V)
        @inbounds for j = 1:obj.J
            Vnew[:, j] = view(V, :, j) .+ K * view(Θ, :, j)
        end
    elseif flavor == "stoch"
        𝛆 = MvNormal(Symmetric(Matrix{T}(obj.Σ)))
        Vnew = similar(V)
        @inbounds for j = 1:obj.J
            Vnew[:, j] = view(V, :, j) .+ K * (view(Θ, :, j) .+ rand(𝛆))
        end
    end   

    # push changes
    push!(obj.V, Vnew)
    push!(obj.Ohist, O)
    obj.iters += 1

    return nothing

end


function EKI_run!(obj::EKIObj{T}, params, H::Function, N_iters::TI; flavor::String="vanilla") where {T<:AbstractFloat, TI<:Int}
    Hens = V -> H_ens(params, V, H)
    for _ in 1:N_iters
        EKI_step!(obj, Hens, flavor)
    end
    return obj
end

