export randomProblem, get_regularized, generate_ensemble, generate_Gaussian_noise, EKR,spectralproj, getLimitMisfits, misfits, misfitPair, getMisfits, stateMisfits, obsMisfits

function randomProblem(n,d,J)
    # Constructs random problem with ensemble having components in both ker(H) and ran(Hᵀ)
    L = 0.5*rand(n,n)
    Σ = L*L' # covariance
    H = rand(n,d) # forward operator
    v₀ = rand(d, J) # ensemble
    v = rand(d,1)
    v /= norm(v)
    H -= H*(v*v') # v ∈ ker(H)
    w = rand(n,1)
    w /= norm(w)
    H -= (w*w')*H # w ∈ ker(Hᵀ)
    basisH,~,~ = svd(H',full=true)
    basisV = hcat(v, basisH[:,2:d-2])
    v₀ = basisV*rand(d-2,J)
    return H,Σ,v₀
end

function generate_Gaussian_noise(J, Γ)
    # Samples Gaussian noise from 𝒩(0, Γ)
    # J = ensemble size
    n = size(Γ, 2)
    μ = vec(zeros(n, 1))
    noiseD = MvNormal(μ, Γ)
    noise = rand(noiseD, J)
    return noise
end

function generate_ensemble(J, μ₀, Γ₀)
    # Samples ensemble from 𝒩(μ₀, Γ₀)
    # J = ensemble size
    prior = MvNormal(μ₀, Γ₀)
    ensemble = rand(prior, J)
    return ensemble
end

function add_noise(M, ε)
    # adds particle noise to all measurements
    k, n = size(ε)
    z = zeros(k ,n)
    for i = 1 : n
        z[:, i] = M + ε[:, i]
    end
    return z
end

function get_regularized(m, Γ₀, Γ, G, μ₀)
    # constructs regularized operators
    z = [m; μ₀]
    F = [G; I]
    Σ = BlockDiagonal([Γ, Γ₀])
    return z, F, Σ

end


function sample_covariance(data::Matrix{Float64})
    n = size(data, 2) 
    mean_vector = mean(data, dims=2)
    centered_data = data .- mean_vector 
    covariance_matrix = (centered_data * centered_data') / (n - 1) 
    return covariance_matrix
end


function EKR(y,H,J,v₀,Σ,steps; 
    method::String="adjfree",
    store_ens::Bool=true)
    # EKRMLE algorithm for a linear H
    n,d = size(H)
    ε = generate_Gaussian_noise(J, Σ)
    z = add_noise(y, ε)
    # allocate memory depending on store_ens argument
    if store_ens
        V = zeros(d,J,steps)
        Γ_store = zeros(d,d,steps)
        V[:,:,1] = v₀
        Γ_store[:,:,1] = cov(v₀')
    else
        means = zeros(d,steps)
        Vnow = v₀
        Γ = cov(v₀')
        means[:,1] = vec(mean(Vnow,dims=2))
    end

    for i = 2 : steps
        if method == "adj"
            Vᵢ = store_ens ? V[:,:,i-1] : Vnow
            μᵢ = mean(Vᵢ,dims=2)
            Γᵢ = (Vᵢ .-μᵢ)*(Vᵢ .-μᵢ)'/(J-1)
            Sᵢ = H*Γᵢ*H' + Σ
            Kᵢ = (Γᵢ*H')/Sᵢ
            Vnext = (I-Kᵢ*H)*Vᵢ 
            Vnext += Kᵢ*z
            if store_ens
                V[:,:,i] = Vnext
                Γ_store[:,:,i] = Γᵢ
            else
                means[:,i] = vec(mean(Vnext,dims=2))
                Γ = Γᵢ
                Vnow = Vnext
            end
        elseif method == "adjfree"
            Vᵢ = store_ens ? V[:,:,i-1] : Vnow
            𝒪 = H*Vᵢ
            HΓᵢHᵀ = sample_covariance(𝒪)
            Γᵢ = sample_covariance(Vᵢ)
            ΓᵢHᵀ = cov(Vᵢ',𝒪')
            Sᵢ = HΓᵢHᵀ + Σ
            Kᵢ = ΓᵢHᵀ/Sᵢ
            Vnext = (I-Kᵢ*H)*Vᵢ
            Vnext += Kᵢ*z
            if store_ens
                V[:,:,i] = Vnext
                Γ_store[:,:,i] = Γᵢ
            else
                means[:,i] = vec(mean(Vnext,dims=2))
                Γ = Γᵢ
                Vnow = Vnext
            end
        end
    end
    if store_ens
        return V,Γ_store,z
    else
        return means,Γ,z
    end
end


function spectralproj(H,v₀,Σ)
    n,d = size(H)
    Γ = cov(v₀')
    HGH = H*Γ*H'
    Hessian = H'*(Σ\H)
    H⁺ = pinv(Hessian)*(H'/Σ)
    # Observation space
    Λ,W = eigen(HGH,Σ) # solve GEV
    Λ = Λ[end:-1:1] # sort in descending order
    W = W[:,end:-1:1] # sort in descending order
    r = sum(broadcast(abs,Λ) .> 1e-10) # number of nonzero λ
    h = rank(H) # rank of H

    # normalize first r eigenvectors
    for i = 1 : r
        W[:,i] /= sqrt(W[:,i]'Σ*W[:,i])
    end

    basisSH = qr(Σ\H).Q # basis of range inv(Σ)H
    basiskerHT = nullspace(H') # basis of ker(Hᵀ)

    v = rand(n,n-r)
    for ℓ = r+1 : n
        if ℓ <= h
            w = basisSH*basisSH'v[:,ℓ-r]
        else
            w = basiskerHT*basiskerHT'*v[:,ℓ-r]
        end
        for k = 1:ℓ-1
            w = w - ((w'Σ*W[:,k])/(W[:,k]'Σ*W[:,k]))*W[:,k]
        end
        W[:,ℓ] = w/sqrt(w'Σ*w)
    end

    𝒫 = Σ*W[:,1:r]*W[:,1:r]'
    if h > r
        𝒬 = Σ*W[:,r+1:h]*W[:,r+1:h]'
    else
        𝒬 = zeros(n,n)
    end
    𝒮 = I-𝒫-𝒬

    # State space
    Λ,U = eigen(Γ*Hessian) # solve GEV
    Λ = Λ[end:-1:1] # sort in descending order
    U = U[:,end:-1:1] # sort in descending order
    # normalize first r eigenvectors
    for i = 1 : r
        U[:,i] /= sqrt(U[:,i]'Hessian*U[:,i])
    end
    #v = rand(n,n-r)
    for ℓ = r+1 : h
            w = basisSH*basisSH'v[:,ℓ-r]
            u = H⁺*Σ*w
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

    return 𝒫,𝒬,𝒮,ℙ,ℚ,𝕊

end

function obsMisfits(V,H,ỹ)
    # Computes all observation misfits
    n,~ = size(H)
    ~,J,iters = size(V)
    Θ = zeros(n,J,iters)
    for i = 1 : iters
        Θ[:,:,i] = H*V[:,:,i] - ỹ
    end
    return Θ
end

function stateMisfits(V,H⁺,ỹ)
    # Computes all state misfits
    d,J,iters = size(V)
    Ω = zeros(d,J,iters)
    Hy = H⁺*ỹ
    for i = 1 : iters
        Ω[:,:,i] = V[:,:,i] - Hy
    end
    return Ω
end

struct misfitPair{T<:AbstractArray}
    mean::T
    cov::T
end

struct misfits{T<:AbstractArray} 
    scrP::misfitPair{T} # 𝒫
    #scrQ::misfitPair{T} # 𝒬
    scrS::misfitPair{T} # 𝒮
    bbP::misfitPair{T} # ℙ
    #bbQ::misfitPair{T} # ℚ
    bbS::misfitPair{T} # 𝕊
end


function getLimitMisfits(ỹ,V,iters,H,Σ,Γ,Hₚ,𝒫,𝒮,ℙ,𝕊)
    # Computes all misfits and stores them in a misfits struct
    𝒫norms = zeros(iters,1); 𝒫covnorms = zeros(iters,1)
    𝒮norms = zeros(iters,1); 𝒮covnorms = zeros(iters,1)
    ℙnorms = zeros(iters,1); ℙcovnorms = zeros(iters,1)
    𝕊norms = zeros(iters,1); 𝕊covnorms = zeros(iters,1)
    H⁺ = Hₚ*((H')/Σ) # weighted pseudoinverse
    v₀ = V[:,:,1]
    h₀ = H*v₀
    v_inf = ℙ*H⁺*ỹ + 𝕊*v₀
    h_inf = 𝒫*ỹ + 𝒮*h₀
    Γ_inf = cov(v_inf')
    HGH_inf = cov(h_inf')

    𝒫hinf = 𝒫*h_inf
    𝒫HGH = 𝒫*HGH_inf*𝒫'
    𝒫HGHnorm = opnorm(𝒫HGH)
    𝒫hnorm = norm(mean(𝒫hinf,dims=2))

    𝒮hinf = 𝒮*h_inf
    𝒮norm = norm(mean(𝒮*h₀ - 𝒮hinf,dims=2))
    𝒮covnorm = opnorm(𝒮*H*Γ[:,:,1]*H'𝒮'-𝒮*HGH_inf*𝒮')

    ℙvinf = ℙ*v_inf
    ℙΓ = ℙ*Γ_inf*ℙ'
    ℙΓnorm = opnorm(ℙΓ)
    ℙvnorm = norm(mean(ℙvinf,dims=2))


    𝕊norm = norm(mean(𝕊*v₀ - 𝕊*v_inf,dims=2))
    𝕊covnorm = opnorm(𝕊*Γ[:,:,1]*𝕊'-𝕊*Γ_inf*𝕊')


    for i = 1 : iters
        hᵢ = H*V[:,:,i]
        vᵢ = V[:,:,i]
        𝒫norms[i] = norm(mean(𝒫*hᵢ - 𝒫hinf,dims=2))/𝒫hnorm
        𝒫covnorms[i] = opnorm(𝒫*H*Γ[:,:,i]*H'𝒫'-
                                    𝒫HGH)/𝒫HGHnorm
        𝒮norms[i] = 𝒮norm
        𝒮covnorms[i] = 𝒮covnorm
        ℙnorms[i] = norm(mean(ℙ*vᵢ - ℙvinf,dims=2))/ℙvnorm
        ℙcovnorms[i] = opnorm(ℙ*Γ[:,:,i]*ℙ'-
                                ℙΓ)/ℙΓnorm
        𝕊norms[i] = 𝕊norm
        𝕊covnorms[i] = 𝕊covnorm
    end

    return misfits(
        misfitPair(𝒫norms,𝒫covnorms),
        misfitPair(𝒮norms,𝒮covnorms),
        misfitPair(ℙnorms,ℙcovnorms),
        misfitPair(𝕊norms,𝕊covnorms),
    )
    
end

