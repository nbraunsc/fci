using PyCall
using Plots 
using LinearAlgebra
using Printf
using Revise

function lanczos(matrix, b, m=12)
    #initalize empty matrices for T and V
    T = zeros(Float64, m+1, m)
    V = zeros(Float64, size(matrix)[1], m+1)
    #normalize start vector
    V[:,1] = b/norm(b)
    #next vector
    w = matrix*V[:,1]
    #orthogonalise
    T[1,1] = dot(w,V[:,1])
    @printf("eigenvalue %f \n", T[1,1])
    w = w - T[1,1]*V[:,1]
    #normalze next vector
    T[2,1] = norm(w)
    V[:,2] = w/T[2,1]
    for j = 2:m
        #make T symmetric
        T[j-1, j] = T[j, j-1]
        #next vector
        w = matrix*V[:,j]
        #orthogonalise agaisnt two previous vectors
        w = w - T[j-1, j]*V[:, j-1] #subtract projection on v_(j-1)
        T[j,j] = dot(w,V[:, j])
        w = w - T[j,j]*V[:,j] #subtract projection on v_j
        #normalize
        T[j+1, j] = norm(w)
        #println(T[j+1, j])
        V[:,j+1] = w/T[j+1, j]
    end

    #make T into symmetric matrix of shape (m,m)
    Tm = T[1:m, 1:m]

    #confirm Lanczos decomposition
    e = zeros(Float64, m)
    e[m] = 1
    #T[m, m-1]
    #value = norm(matrix*V[:,1:m] - V[:,1:m]*Tm) - T[m, m-1]*(dot(V[:,1:m], e))
    #value
    return Tm, V
end
#={{{=#
#n = ARGS[1]
#m = ARGS[2]
#before = rand(Float64, (n,n))
#matrix = Symmetric(before)
#Tm, V = lanczos(matrix, rand(Float64, n),m)
#println("\nTm matrix from Lanczos:")
#Tm}}}
