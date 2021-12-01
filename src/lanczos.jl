using PyCall
using Plots 
using LinearAlgebra
using Printf
using Revise

#n = 500
#m = 12
#before = rand(Float64, (n,n))
#matrix = Symmetric(before)
#Tm, V = lanczos(matrix, rand(Float64, n),m)

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
    w = w - T[1,1]*V[:,1]
    #normalze next vector
    T[2,1] = norm(w)
    V[:,2] = w/T[2,1]

    for j = 2:m
        #make T symmetric
        println(T[j, j-1])
        
        T[j-1, j] = T[j, j-1]
        
        #next vector
        w = matrix*V[:,j]
        #orthogonalise agaisnt two previous vectors
        T[j,j] = dot(w,V[:, j])
        # below is like this w_3 = w_3 - a_2 |v_2> - b_2 |v_1>
        w = w - dot(w,V[:,j])*V[:,j] - T[j-1, j]*V[:, j-1] #subtract projection on v_j and v_(j-1) 
        
        # note: <v_2 | H | v_0> = 0 so only have to subtract two previous vectors for othogonalization
        # that is what is being checked below
        #if j > 3
        #    value = dot(w, V[:,j-2])
        #    println("\n overlap with diff >1: ", value)
        #end
        
        #normalize
        T[j+1, j] = norm(w)
        V[:,j+1] = w/T[j+1, j]

        #convergence check
        if T[j+1, j] < 10E-8
            @printf("\n\nConverged at %i iteration \n", j)
            print("\n", T[j+1, j])
            Tm = T[1:j, 1:j]
            return Tm, V
            break
        end 
    end

    #make T into symmetric matrix of shape (m,m)
    Tm = T[1:m, 1:m]
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
