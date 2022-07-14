using fci
using LinearAlgebra
using JLD2
using Printf
using StaticArrays

#function orthogonalise(V::Matrix{Float64}, T::Matrix{Float64},  w::Vector{Float64}, k)
#    h_k = V[:,1:k]'*w
#    w_new = w - V[:,1:k]*h_k
#    return w_new
#end

function make_t(alpha, beta)
    shape_t = size(alpha)[1]
    T = zeros(shape_t, shape_t)
    for i = 1:shape_t-1
        T[i,i] = alpha[i]
        T[i,i+1] = beta[i+1]
        T[i+1,i] = beta[i+1]
    end
    T[shape_t, shape_t] = alpha[shape_t]
    return T
end


function Lanczos(p::FCIProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)
    eigval = 0.0
    
    if ci_vector == nothing
        v = rand(p.dim)
        #S = v'*v
        #v = v*inv(sqrt(S))
    else
        v = vec(ci_vector)
        #S = v'*v
        #v = v*inv(sqrt(S))
    end

    alpha = zeros(Float64, max_iter)
    beta = zeros(Float64, max_iter-1)
    
    org = deepcopy(v/norm(v)) 
    beta[1] = norm(v)
    r = v
    T = zeros(Float64, max_iter+1, max_iter)
    V = zeros(Float64, p.dim, max_iter+1)

    for j in 1:max_iter
        if j == 1
            V[:,j] = r/beta[1]
        else
            V[:,j] = r/beta[j-1]
        end

        r = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, V[:,j]) 

        if j != 1
            r = r - V[:,j-1]*beta[j-1]
        end
        alpha[j] = V[:,j]'*r
        r = r - V[:,j]*alpha[j]

        if j != 1
            #reorthogonalization
            for k in 1:j-1
                qk = V[:,k]
                x = r'*qk
                r = r - x*qk
                #check ortho
                check = r'*V[:,k]
                if check > 1e-14
                    error("not orthogonal")
                end

            end
            beta[j] = norm(r)
            
            #Fill T matrix
            T = make_t(alpha[1:j], beta[1:j])
            #Base.display(T)

            #compute approx eigenvlaes of Tj
            F = eigen(T)
            theta_i = F.values[1]
            s_i = F.vectors[:,1]
            res = abs(beta[j]*s_i[j])
            println("  Residual: ", res)

            ritz_vec = V[:,1:j]*s_i
            Ax = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, ritz_vec)
            other_res = norm(Ax - ritz_vec*theta_i)
            #other_res = norm(Ax - org)
            #println(" OTHER Residual: ", other_res)



            if res <= tol
                println("\n ----------- CONVERGED in ", j, " iterations -------------\n ")
                eval = F.values[1]
                vector = V[:,1:j]*F.vectors[:,j]
                return eval
            end
        end
    end
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
    println(norm(b))
    error("here")
    q1 = b/norm(b)
    V[:,1] = q1
    #next vector
    w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, Ha, Hb, q1) 
    #orthogonalise
    a1 = dot(w,V[:,1])
    T[1,1] = a1
    w = w - a1*q1
    #normalize next vector
    bk = norm(w)
    T[2,1] = bk
    V[:,2] = w/bk
    #wk = w

    for i = 2:max_iter
        qk = V[:,i]
        qk_1 = V[:,i-1]

        #make T symmetric by setting b values across diagonal
        T[i-1, i] = T[i, i-1]
        
        #next vector
        wk = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, Ha, Hb, qk) 
        #res = norm(w - wk)
        #res = norm(w - q1'*wk*q1)
        #println(res)
        
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

        bk = norm(wk)
        T[i+1, i] = bk
        qk = wk/bk
        V[:,i+1] = qk
        
        Tm = T[1:i-1, 1:i-1]
        Vm = V[:, 1:i-1]
        eig = eigen(Tm)
        ritz_val = eig.values[1]
        ritz_vec = Vm*eig.vectors[:,1]
        e = ones(size(ritz_vec))
        res = bk*abs(e'*ritz_vec)
        println(res)
        
        if res <= tol 
            println("\n ----------- CONVERGED in ", i, " iterations -------------\n ")
            wk = wk - bk*qk_1
            ak = wk'*qk
            T[i,i] = ak
            Base.display(T[1:i, 1:i])
            Tm = T[1:i, 1:i]
            eig = eigen(Tm)
            eigval = eig.values[1]
            return eigval
        end
        
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

#function Lanczos(p::FCIProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)
#    eigval = 0.0
#    
#    if ci_vector == nothing
#        b = rand(p.dim)
#        S = b'*b
#        b = b*inv(sqrt(S))
#    else
#        b = vec(ci_vector)
#        S = b'*b
#        b = b*inv(sqrt(S))
#    end
#    
#    T = zeros(Float64, max_iter+1, max_iter)
#    V = zeros(Float64, p.dim, max_iter+1)
#    #normalize start vector
#    q1 = b/norm(b)
#    V[:,1] = q1
#    #next vector
#    w = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, b) 
#    #orthogonalise
#    a1 = dot(w,V[:,1])
#    T[1,1] = a1
#    w = w - a1*q1
#    #normalize next vector
#    b2 = norm(w)
#    T[2,1] = b2
#    V[:,2] = w/b2
#    bk = norm(w)
#    wk = w
#
#    for i = 2:max_iter
#        qk = V[:,i]
#        qk_1 = V[:,i-1]
#
#        #make T symmetric by setting b values across diagonal
#        T[i-1, i] = T[i, i-1]
#        
#        #next vector
#        wk = fci.matvec(a_configs, b_configs, a_lookup, b_lookup, ints, p, qk) 
#        wk = wk - bk*qk_1
#        ak = wk'*qk
#        T[i,i] = ak
#        wk = wk - ak*qk
#
#        #reorthogonalization
#        vec = zeros(size(wk))
#        for j in 1:i-1
#            qj = V[:,j]
#            x = wk'*qj
#            vec = vec + x*qj
#        end
#        wk = wk-vec
#
#        #check point
#        bk = norm(wk)
#        if bk <= 1e-12
#            Tm = T[1:i, 1:i]#=}}}=#
#            eig = eigen(Tm)
#            eigval = eig.values[1]
#            return eigval
#        end
#        T[i+1, i] = bk
#        qk = wk/bk
#        V[:,i+1] = qk
#        
#        if i == max_iter
#            println("\n ----------- HIT MAX ITERATIONS -------------\n ")
#            #make T into symmetric matrix of shape (m,m)
#            Tm = T[1:max_iter, 1:max_iter]#=}}}=#
#            eig = eigen(Tm)
#            eigval = eig.values[1]
#            #println(eigvals)
#        end
#    end
#    return eigval
#end

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
