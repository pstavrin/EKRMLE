export getData, getNoisy, getGamPr, getG, solveHE

function getData()
    data = matread("data/heat-cont.mat")
    A, C = data["A"], data["C"]
    return A,C
end

function getGamPr(A,B)
    # solves Lyapunov equation AΓ + ΓAᵀ + BBᵀ = 0, enforces symmetricity
    # Γ is the compatible prior
    Γ_pr = lyap(Matrix(A),Matrix(B*B')) # prior cov
    d = size(Γ_pr,2)
    for j = 1 : d
        for i = j+1 : d
            Γ_pr[i, j] = Γ_pr'[i, j] # enforce symmetricity
        end
    end
    return Γ_pr
end

function solveHE(A,C,X₀,Δt,T)
    # Solves HE using Forward Euler
    nₜ = length(T) # number of time steps
    Y = zeros(nₜ,1) # solution vector
    Y[1] = (C*X₀)[1]
    for i = 2 : nₜ
        X₀ += Δt*(A*X₀)
        Y[i] = (C*X₀)[1]
    end
    return Y
end

function getG(A,C,h,Δt,n)
    # Constructs forward operator G explicitly
    d = size(A,2)
    skips = Int(round(h/Δt))
    M = I + Δt*A
    Mks = M^skips
    G = zeros(n,d)
    Mpow = I
    for i = 1 : n
        Mpow = Mks*Mpow
        G[i,:] = vec(C*Mpow)'
    end
    return G
end

function getNoisy(Y,Γ_obs,h,Δt)
    # gets noisy measurements
    nₜ = length(Y)
    skips = Int(round(h/Δt))
    idx = 1+skips:skips:nₜ # measurement indices
    sampleY = Y[idx] # sample solution at indices
    noise = generate_Gaussian_noise(1,Γ_obs)
    m = sampleY + noise
end