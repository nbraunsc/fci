using fci
using LinearAlgebra
using JLD2
using Printf
using StaticArrays

function Lanczos(dim, norbs, nalpha, nbeta, H_alpha, H_beta, Ia, Ib, alpha_configs, beta_configs, int2e, index_table_a, index_table_b, ndets_a, sign_table_a, sign_table_b, max_iter::Int=12, ϑ::Float64=1e-8)
    #@load "/Users/nicole/code/fci/test/data/_testdata_h8_integrals.jld2"
    #@load "/Users/nicole/code/fci/test/data/_testdata_h8.jld2"
    #@load "/Users/nicole/code/fci/test/data/_testdata_h8_signs.jld2"

    #b = rand(dim)
    b = zeros(Float64, dim)
    b[1] = 1
    
    #initalize empty matrices for T and V
    T = zeros(Float64, max_iter+1, max_iter)
    V = zeros(Float64, dim, max_iter+1)
    #normalize start vector
    V[:,1] = b/norm(b)
    #next vector
    w = fci.get_sigma(H_alpha, H_beta, Ia, Ib, V[:,1], [alpha_configs, beta_configs], norbs, int2e, index_table_a, index_table_b, dim, ndets_a, sign_table_a, sign_table_b)
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
        w = fci.get_sigma(H_alpha, H_beta, Ia, Ib, V[:,j], [alpha_configs, beta_configs], norbs, int2e, index_table_a, index_table_b, dim, ndets_a, sign_table_a, sign_table_b)
        #orthogonalise agaisnt two previous vectors
        T[j,j] = dot(w,V[:, j])
        # below is like this w_3 = w_3 - a_2 |v_2> - b_2 |v_1>
        w = w - dot(w,V[:,j])*V[:,j] - T[j-1, j]*V[:, j-1] #subtract projection on v_j and v_(j-1) 
        
        # note: <v_2 | H | v_0> = 0 so only have to subtract two previous vectors for othogonalization
        #normalize
        T[j+1, j] = norm(w)
        V[:,j+1] = w/T[j+1, j]

        #convergence check
        if T[j+1, j] < ϑ
            @printf("\n --------------- Converged at %i iteration ---------------  \n", j)
            Tm = T[1:j, 1:j]
            eig = eigen(Tm)
            eigvals = eig.values[1]
            println(eigvals)
        end 

        if j == max_iter
            println("\n ----------- HIT MAX ITERATIONS -------------\n ")
            #make T into symmetric matrix of shape (m,m)
            Tm = T[1:max_iter, 1:max_iter]#=}}}=#
            eig = eigen(Tm)
            eigvals = eig.values[1]
            println(eigvals)
        end
    end
end
