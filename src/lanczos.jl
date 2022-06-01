using fci
using LinearAlgebra
using JLD2
using Printf
using StaticArrays

function orthogonalise(V::Matrix{Float64}, T::Matrix{Float64},  w::Vector{Float64}, k)
    h_k = V[:,1:k]'*w
    w_new = w - V[:,1:k]*h_k
    return w_new
end

function Lanczos(p::FCIProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, Ha, Hb, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)
    eigval = 0.0
    
    if ci_vector == nothing
        b = rand(p.dim)
        S = b'*b
        b = b*inv(sqrt(S))
    else
        b = vec(ci_vector)
        S = b'*b
        b = b*inv(sqrt(S))
    end
    
    T = zeros(Float64, max_iter+1, max_iter)
    V = zeros(Float64, p.dim, max_iter+1)
    #normalize start vector
    q1 = b/norm(b)
    V[:,1] = q1
    #next vector
    w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, Ha, Hb, b) 
    #orthogonalise
    a1 = dot(w,V[:,1])
    T[1,1] = a1
    w = w - a1*q1
    #normalize next vector
    b2 = norm(w)
    T[2,1] = b2
    V[:,2] = w/b2
    bk = norm(w)
    wk = w

    for i = 2:max_iter
        qk = V[:,i]
        qk_1 = V[:,i-1]

        #make T symmetric by setting b values across diagonal
        T[i-1, i] = T[i, i-1]
        
        #next vector
        wk = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, Ha, Hb, qk) 
        wk = wk - bk*qk_1
        ak = wk'*qk
        T[i,i] = ak
        wk = wk - ak*qk

        #reorthogonalization
        vec = zeros(size(wk))
        for j in 1:i-1
            qj = V[:,j]
            x = wk'*qj
            vec = vec + x*qj
        end
        wk = wk-vec

        #check point
        bk = norm(wk)
        if bk <= 1e-12
            Tm = T[1:i, 1:i]#=}}}=#
            eig = eigen(Tm)
            eigval = eig.values[1]
            return eigval
        end
        T[i+1, i] = bk
        qk = wk/bk
        V[:,i+1] = qk
        
        if i == max_iter
            println("\n ----------- HIT MAX ITERATIONS -------------\n ")
            #make T into symmetric matrix of shape (m,m)
            Tm = T[1:max_iter, 1:max_iter]#=}}}=#
            eig = eigen(Tm)
            eigval = eig.values[1]
            #println(eigvals)
        end
    end
    return eigval
end

function Lanczos(p::FCIProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)
    eigval = 0.0
    
    if ci_vector == nothing
        b = rand(p.dim)
        S = b'*b
        b = b*inv(sqrt(S))
    else
        b = vec(ci_vector)
        S = b'*b
        b = b*inv(sqrt(S))
    end
    
    T = zeros(Float64, max_iter+1, max_iter)
    V = zeros(Float64, p.dim, max_iter+1)
    #normalize start vector
    q1 = b/norm(b)
    V[:,1] = q1
    #next vector
    w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, b) 
    #orthogonalise
    a1 = dot(w,V[:,1])
    T[1,1] = a1
    w = w - a1*q1
    #normalize next vector
    b2 = norm(w)
    T[2,1] = b2
    V[:,2] = w/b2
    bk = norm(w)
    wk = w

    for i = 2:max_iter
        qk = V[:,i]
        qk_1 = V[:,i-1]

        #make T symmetric by setting b values across diagonal
        T[i-1, i] = T[i, i-1]
        
        #next vector
        wk = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, qk) 
        wk = wk - bk*qk_1
        ak = wk'*qk
        T[i,i] = ak
        wk = wk - ak*qk

        #reorthogonalization
        vec = zeros(size(wk))
        for j in 1:i-1
            qj = V[:,j]
            x = wk'*qj
            vec = vec + x*qj
        end
        wk = wk-vec

        #check point
        bk = norm(wk)
        if bk <= 1e-12
            Tm = T[1:i, 1:i]#=}}}=#
            eig = eigen(Tm)
            eigval = eig.values[1]
            return eigval
        end
        T[i+1, i] = bk
        qk = wk/bk
        V[:,i+1] = qk
        
        if i == max_iter
            println("\n ----------- HIT MAX ITERATIONS -------------\n ")
            #make T into symmetric matrix of shape (m,m)
            Tm = T[1:max_iter, 1:max_iter]#=}}}=#
            eig = eigen(Tm)
            eigval = eig.values[1]
            #println(eigvals)
        end
    end
    return eigval
end

function Lanczos(p::RASProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, Ha, Hb, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)
    eigval = 0.0
    
    if ci_vector == nothing
        b = rand(p.dim)
        S = b'*b
        b = b*inv(sqrt(S))
    else
        b = vec(ci_vector)
        S = b'*b
        b = b*inv(sqrt(S))
    end
    
    T = zeros(Float64, max_iter+1, max_iter)
    V = zeros(Float64, p.dim, max_iter+1)
    #normalize start vector
    q1 = b/norm(b)
    V[:,1] = q1
    #next vector
    w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, Ha, Hb, b) 
    #orthogonalise
    a1 = dot(w,V[:,1])
    T[1,1] = a1
    w = w - a1*q1
    #normalize next vector
    b2 = norm(w)
    T[2,1] = b2
    V[:,2] = w/b2
    bk = norm(w)
    wk = w

    for i = 2:max_iter
        qk = V[:,i]
        qk_1 = V[:,i-1]

        #make T symmetric by setting b values across diagonal
        T[i-1, i] = T[i, i-1]
        
        #next vector
        wk = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, Ha, Hb, qk) 
        wk = wk - bk*qk_1
        ak = wk'*qk
        T[i,i] = ak
        wk = wk - ak*qk

        #reorthogonalization
        vec = zeros(size(wk))
        for j in 1:i-1
            qj = V[:,j]
            x = wk'*qj
            vec = vec + x*qj
        end
        wk = wk-vec

        #check point
        bk = norm(wk)
        if bk <= 1e-12
            Tm = T[1:i, 1:i]#=}}}=#
            eig = eigen(Tm)
            eigval = eig.values[1]
            return eigval
        end
        T[i+1, i] = bk
        qk = wk/bk
        V[:,i+1] = qk
        
        if i == max_iter
            println("\n ----------- HIT MAX ITERATIONS -------------\n ")
            #make T into symmetric matrix of shape (m,m)
            Tm = T[1:max_iter, 1:max_iter]#=}}}=#
            eig = eigen(Tm)
            eigval = eig.values[1]
            #println(eigvals)
        end
    end
    return eigval
end

function Lanczos(p::RASProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)
    eigval = 0.0
    
    if ci_vector == nothing
        b = rand(p.dim)
        S = b'*b
        b = b*inv(sqrt(S))
    else
        b = vec(ci_vector)
        S = b'*b
        b = b*inv(sqrt(S))
    end
    
    T = zeros(Float64, max_iter+1, max_iter)
    V = zeros(Float64, p.dim, max_iter+1)
    #normalize start vector
    q1 = b/norm(b)
    V[:,1] = q1
    #next vector
    w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, b) 
    #orthogonalise
    a1 = dot(w,V[:,1])
    T[1,1] = a1
    w = w - a1*q1
    #normalize next vector
    b2 = norm(w)
    T[2,1] = b2
    V[:,2] = w/b2
    bk = norm(w)
    wk = w

    for i = 2:max_iter
        qk = V[:,i]
        qk_1 = V[:,i-1]

        #make T symmetric by setting b values across diagonal
        T[i-1, i] = T[i, i-1]
        
        #next vector
        wk = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, qk) 
        wk = wk - bk*qk_1
        ak = wk'*qk
        T[i,i] = ak
        wk = wk - ak*qk

        #reorthogonalization
        vec = zeros(size(wk))
        for j in 1:i-1
            qj = V[:,j]
            x = wk'*qj
            vec = vec + x*qj
        end
        wk = wk-vec

        #check point
        bk = norm(wk)
        if bk <= 1e-12
            Tm = T[1:i, 1:i]#=}}}=#
            eig = eigen(Tm)
            eigval = eig.values[1]
            return eigval
        end
        T[i+1, i] = bk
        qk = wk/bk
        V[:,i+1] = qk
        
        if i == max_iter
            println("\n ----------- HIT MAX ITERATIONS -------------\n ")
            #make T into symmetric matrix of shape (m,m)
            Tm = T[1:max_iter, 1:max_iter]#=}}}=#
            eig = eigen(Tm)
            eigval = eig.values[1]
            #println(eigvals)
        end
    end
    return eigval
end


        

function old_Lanczos(p::FCIProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, Ha, Hb, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)#={{{=#
    eigval = 0.0
    
    if ci_vector == nothing
        b = rand(p.dim)
        S = b'*b
        b = b*inv(sqrt(S))
    else
        b = vec(ci_vector)
        S = b'*b
        b = b*inv(sqrt(S))
    end
    
    T = zeros(Float64, max_iter+1, max_iter)
    V = zeros(Float64, p.dim, max_iter+1)
    #normalize start vector
    V[:,1] = b/norm(b)
    #next vector
    w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, Ha, Hb, b)
    #orthogonalise
    T[1,1] = dot(w,V[:,1])
    w = w - T[1,1]*V[:,1]
    #normalize next vector
    T[2,1] = norm(w)
    V[:,2] = w/T[2,1]

    for j = 2:max_iter
        #make T symmetric
        T[j-1, j] = T[j, j-1]
        
        #next vector
        w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, Ha, Hb, V[:,j])
        
        #need to orthogonalise agaisnt all previous vectors in V
        T[j,j] = dot(w,V[:, j])
        w = orthogonalise(V, T,  w, j)

        #Check point
        if norm(w) <= 1e-12
            println("Norm is zero!")
            error("here")
        end

        #normalize
        T[j+1, j] = norm(w)
        V[:,j+1] = w/T[j+1, j]

        #residual to check convergence, residual not correct
        res = V[:,j]- fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, Ha, Hb, V[:,j+1])
        if norm(res) <= tol
            println("\n ----------- CONVERGED -------------\n ")
            Tm = T[1:j, 1:j]
            eig = eigen(Tm)
            eigval = eig.values[1]
            return eigval
        end
        
        if j == max_iter
            println("\n ----------- HIT MAX ITERATIONS -------------\n ")
            #make T into symmetric matrix of shape (m,m)
            Tm = T[1:max_iter, 1:max_iter]
            eig = eigen(Tm)
            eigval = eig.values[1]
            #println(eigvals)
        end
    end
    return eigval
end

function old_Lanczos(p::FCIProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)
    eigval = Float64

    if ci_vector == nothing
        b = rand(p.dim)
        S = b'*b
        b = b*inv(sqrt(S))
    else
        b = vec(ci_vector)
        S = b'*b
        b = b*inv(sqrt(S))
    end
    
    T = zeros(Float64, max_iter+1, max_iter)
    V = zeros(Float64, p.dim, max_iter+1)
    #normalize start vector
    V[:,1] = vec(b)/norm(vec(b))
    #next vector
    w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, b)
    #orthogonalise
    T[1,1] = dot(w,V[:,1])
    w = w - T[1,1]*V[:,1]
    #normalize next vector
    T[2,1] = norm(w)
    V[:,2] = w/T[2,1]

    for j = 2:max_iter
        #make T symmetric
        T[j-1, j] = T[j, j-1]
        
        #next vector
        w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, V[:,j])
        #need to orthogonalise agaisnt all previous vectors in V
        T[j,j] = dot(w,V[:, j])
        w = orthogonalise(V, T,  w, j)
        
        #normalize
        T[j+1, j] = norm(w)
        V[:,j+1] = w/T[j+1, j]
        
        if j == max_iter
            #println("\n ----------- HIT MAX ITERATIONS -------------\n ")
            #make T into symmetric matrix of shape (m,m)
            Tm = T[1:max_iter, 1:max_iter]#=}}}=#
            eig = eigen(Tm)#={{{=#
            eigval = eig.values[1]
            #println(eigvals)
        end
    end
    return eigval
end

function old_Lanczos(p::RASProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, Ha, Hb, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)
    eigval = 0.0
    if ci_vector == nothing
        b = rand(p.dim)
        S = b'*b
        b = b*inv(sqrt(S))
    else
        b = vec(ci_vector)
        S = b'*b
        b = b*inv(sqrt(S))
    end
    
    T = zeros(Float64, max_iter+1, max_iter)
    V = zeros(Float64, p.dim, max_iter+1)
    #normalize start vector
    V[:,1] = b/norm(b)
    #next vector
    w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, Ha, Hb, b)
    #orthogonalise
    T[1,1] = dot(w,V[:,1])
    w = w - T[1,1]*V[:,1]
    #normalize next vector
    T[2,1] = norm(w)
    V[:,2] = w/T[2,1]

    for j = 2:max_iter
        #make T symmetric
        T[j-1, j] = T[j, j-1]
        
        #next vector
        w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, Ha, Hb, V[:,j])
        #need to orthogonalise agaisnt all previous vectors in V
        T[j,j] = dot(w,V[:, j])
        w = orthogonalise(V, T,  w, j)
        
        #normalize
        T[j+1, j] = norm(w)
        V[:,j+1] = w/T[j+1, j]
        
        if j == max_iter
            #println("\n ----------- HIT MAX ITERATIONS -------------\n ")
            #make T into symmetric matrix of shape (m,m)
            Tm = T[1:max_iter, 1:max_iter]#=}}}=#
            eig = eigen(Tm)#={{{=#
            eigval = eig.values[1]
            #println(eigvals)
        end
    end
    return eigval
end

function old_Lanczos(p::RASProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)
    eigval = Float64

    if ci_vector == nothing
        b = rand(p.dim)
        S = b'*b
        b = b*inv(sqrt(S))
    else
        b = vec(ci_vector)
        S = b'*b
        b = b*inv(sqrt(S))
    end
    
    T = zeros(Float64, max_iter+1, max_iter)
    V = zeros(Float64, p.dim, max_iter+1)
    #normalize start vector
    V[:,1] = vec(b)/norm(vec(b))
    #next vector
    w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, b)
    #orthogonalise
    T[1,1] = dot(w,V[:,1])
    w = w - T[1,1]*V[:,1]
    #normalize next vector
    T[2,1] = norm(w)
    V[:,2] = w/T[2,1]

    for j = 2:max_iter
        #make T symmetric
        T[j-1, j] = T[j, j-1]
        
        #next vector
        w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, V[:,j])
        #need to orthogonalise agaisnt all previous vectors in V
        T[j,j] = dot(w,V[:, j])
        w = orthogonalise(V, T,  w, j)
        #normalize
        T[j+1, j] = norm(w)
        V[:,j+1] = w/T[j+1, j]
        
        if j == max_iter
            #println("\n ----------- HIT MAX ITERATIONS -------------\n ")
            #make T into symmetric matrix of shape (m,m)
            Tm = T[1:max_iter, 1:max_iter]#=}}}=#
            eig = eigen(Tm)#={{{=#
            eigval = eig.values[1]
            #println(eigvals)
        end
    end
    return eigval
end#=}}}=#
