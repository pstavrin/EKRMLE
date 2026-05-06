export randomLinearProblemObj, spectralproj

mutable struct randomLinearProblemObj{T<:AbstractFloat,TI<:Int}
    # Problem dimensions
    n::TI
    d::TI
    J::TI

    # Operators
    H::Matrix{T}
    Γ::Matrix{T}
    Hess::Matrix{T}
    pHess::Matrix{T}
    H⁺::Matrix{T}

    # data
    y::Vector{T}
    truth::Vector{T}

    # solutions
    v_star::Vector{T}

    # Initial ensemble
    V0::Matrix{T}
    
end


function randomLinearProblemObj(n::TI, d::TI, J::TI; rankdef::Bool=true, T=Float64, rng=Random.default_rng()) where {TI<:Int}
    # Random forward opertor
    H = rand(rng, T, n, d)
    L = (0.5)*rand(rng, T, n, n)
    Γ = Symmetric(L * L' + 1e-1*I)


    if rankdef
        v = rand(d,1)
        v /= norm(v)
        H -= H*(v*v') # v ∈ ker(H)
        w = rand(n,1)
        w /= norm(w)
        H -= (w*w')*H # w ∈ ker(Hᵀ)
        basisH,~,~ = svd(H',full=true)
        basisV = hcat(v, basisH[:,2:d-2])
        V₀ = basisV*rand(d-2,J)
    else
        V₀ = rand(rng, T, d, J)
    end


    truth = rand(d, 1)
    ε = rand(rng, MvNormal(Γ)) # noise
    y = H*truth + ε # noisy data
    y = vec(y)
    Hess = H'*(Γ\H)
    pHessian = pinv(Hess)
    H⁺ = pHessian*((H')/Γ) # weighted pseudoinverse
    v_star = H⁺ * y

    randomLinearProblemObj{T,TI}(
        n, d, J,
        Matrix{T}(H), Matrix{T}(Γ), Matrix{T}(Hess), Matrix{T}(pHessian), Matrix{T}(H⁺),
        y, vec(truth), vec(v_star),
        Matrix{T}(V₀)
    )
end




mutable struct spectralproj{T<:AbstractFloat}
    P::Matrix{T}
    S::Matrix{T}
    calP::Matrix{T}
    calS::Matrix{T}
end

function spectralproj(prob::randomLinearProblemObj,C::AbstractArray{T}) where {T<:AbstractFloat}
    H = prob.H
    Γ = prob.Γ
    n, d = prob.n, prob.d
    HGH = H*C*H'
    Hessian = prob.Hess
    H⁺ = prob.H⁺
    # Observation space
    Λ,W = eigen(HGH,Γ) # solve GEV
    Λ = Λ[end:-1:1] # sort in descending order
    W = W[:,end:-1:1] # sort in descending order
    r = sum(broadcast(abs,Λ) .> 1e-10) # number of nonzero λ
    h = rank(H) # rank of H

    # normalize first r eigenvectors
    for i = 1 : r
        W[:,i] /= sqrt(W[:,i]'Γ*W[:,i])
    end

    basisSH = qr(Γ\H).Q # basis of range inv(Γ)H
    basiskerHT = nullspace(H') # basis of ker(Hᵀ)

    v = rand(n,n-r)
    for ℓ = r+1 : n
        if ℓ <= h
            w = basisSH*basisSH'v[:,ℓ-r]
        else
            w = basiskerHT*basiskerHT'*v[:,ℓ-r]
        end
        for k = 1:ℓ-1
            w = w - ((w'Γ*W[:,k])/(W[:,k]'Γ*W[:,k]))*W[:,k]
        end
        W[:,ℓ] = w/sqrt(w'Γ*w)
    end

    𝒫 = Γ*W[:,1:r]*W[:,1:r]'
    if h > r
        𝒬 = Γ*W[:,r+1:h]*W[:,r+1:h]'
    else
        𝒬 = zeros(n,n)
    end
    𝒮 = I-𝒫-𝒬

    # State space
    Λ,U = eigen(C*Hessian) # solve GEV
    Λ = Λ[end:-1:1] # sort in descending order
    U = U[:,end:-1:1] # sort in descending order
    # normalize first r eigenvectors
    for i = 1 : r
        U[:,i] /= sqrt(U[:,i]'Hessian*U[:,i])
    end
    #v = rand(n,n-r)
    for ℓ = r+1 : h
            w = basisSH*basisSH'v[:,ℓ-r]
            u = H⁺*Γ*w
        for k = 1:ℓ-1
            u = u - ((u'Hessian*U[:,k])/(U[:,k]'Hessian*U[:,k]))*U[:,k]
        end
        U[:,ℓ] = u/sqrt(u'Hessian*u)
    end

    ℙ = U[:,1:r]*U[:,1:r]'Hessian
    if h > r
        ℚ = U[:,r+1:h]*U[:,r+1:h]'Hessian
    else
        ℚ = zeros(d,d)
    end
    𝕊 = I-ℙ-ℚ

    spectralproj(
        Matrix{T}(ℙ),
        Matrix{T}(I - ℙ),
        Matrix{T}(𝒫),
        Matrix{T}(I - 𝒫)
    )
end