using fci
using LinearAlgebra
using JLD2
using Printf
using StaticArrays

function Lanczos(p::FCIProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, Ha, Hb, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)
    eigval = 0.0
    if ci_vector == nothing
        b = rand(p.dim)
    else
        b = vec(ci_vector)
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
        #orthogonalise agaisnt two previous vectors
        T[j,j] = dot(w,V[:, j])
        w = w - dot(w,V[:,j])*V[:,j] - T[j-1, j]*V[:, j-1] #subtract projection on v_j and v_(j-1) 
        #normalize
        T[j+1, j] = norm(w)
        V[:,j+1] = w/T[j+1, j]
        
        if j == max_iter
            println("\n ----------- HIT MAX ITERATIONS -------------\n ")
            #make T into symmetric matrix of shape (m,m)
            Tm = T[1:max_iter, 1:max_iter]#=}}}=#
            Base.display(Tm)
            eig = eigen(Tm)
            eigval = eig.values[1]
            println(eigvals)
        end
    end
    return eigval
end

function Lanczos(p::FCIProblem, ints::H, a_configs, b_configs, a_lookup, b_lookup, ci_vector=nothing, max_iter::Int=12, tol::Float64=1e-8)
    eigval = Float64

    if ci_vector == nothing
        b = rand(p.dim)
    else
        b = ci_vector
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
        #orthogonalise agaisnt two previous vectors
        T[j,j] = dot(w,V[:, j])
        w = w - dot(w,V[:,j])*V[:,j] - T[j-1, j]*V[:, j-1] #subtract projection on v_j and v_(j-1) 
        #normalize
        T[j+1, j] = norm(w)
        V[:,j+1] = w/T[j+1, j]
        
        if j == max_iter
            println("\n ----------- HIT MAX ITERATIONS -------------\n ")
            #make T into symmetric matrix of shape (m,m)
            Tm = T[1:max_iter, 1:max_iter]#=}}}=#
            eig = eigen(Tm)
            eigval = eig.values[1]
            println(eigvals)
        end
    end
    return eigval
end


    

