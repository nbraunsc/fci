using fci
using LinearAlgebra
using JLD2

function eachcolumn(V::AbstractMatrix)
    [V[:,i] for i in 1:size(V,2)]
end

function orthonormalize(V::Matrix{Float64})
    Q,R = qr(V)
    return Q*Matrix{Float64}(I,size(V)...)
end

function fullmatrix_Davidson(A::AbstractMatrix, NTargetEigvals::Int=1, subspaceincrement::Int=8, maxsubspacesize::Int=200, ϑ::Float64=1e-8)
    k = subspaceincrement
    M = min(maxsubspacesize, size(A,2))
    
    V = Matrix{Float64}(I,size(A,1),k)
    eigvals = ones(NTargetEigvals)
    for m in k:k:M
        V   = orthonormalize(V)
        
        AV  = A * V
        T   = Symmetric(V' * AV)
        eig = eigen(T)
        
        w = [AV*y - λ*(V*y) for (λ,y) in zip(eig.values[1:k],eachcolumn(eig.vectors)[1:k])]
        V = [V w...]
        
        eigvalsN = eig.values[1:NTargetEigvals]
        eigvals  = norm(eigvalsN-eigvals)>=ϑ ? eigvalsN : return eig.values
    end
    error("Davidson did not converge in $maxsubspacesize-dimensional subspace")
end

function Davidson(dim, H_alpha, H_beta, Ia, Ib, alpha_configs, beta_configs, norbs, int2e, index_table_a, index_table_b, ndets_a, sign_table_a, sign_table_b, NTargetEigvals::Int=1, subspaceincrement::Int=8, maxsubspacesize::Int=200, ϑ::Float64=1e-8)
#function Davidson(A::AbstractMatrix, dim, NTargetEigvals::Int=1, subspaceincrement::Int=8, maxsubspacesize::Int=200, ϑ::Float64=1e-8)
    #@load "/Users/nicole/code/fci/test/data/_testdata_h8_integrals.jld2"
    #@load "/Users/nicole/code/fci/test/data/_testdata_h8.jld2"
    k = subspaceincrement
    M = min(maxsubspacesize, dim)
    
    V = Matrix{Float64}(I,dim,k)
    eigvals = ones(NTargetEigvals)
    for m in k:k:M
        V   = orthonormalize(V)
        
        AV = Matrix{Float64}(zeros(Float64, dim, size(V,2)))
        for i in 1:size(V,2)
            x = fci.get_sigma(H_alpha, H_beta, Ia, Ib, V[:,i], [alpha_configs, beta_configs], norbs, int2e, index_table_a, index_table_b, dim, ndets_a, sign_table_a, sign_table_b)
            AV[:,i] = x
        end
        
        T   = Symmetric(V' * AV)
        eig = eigen(T)
        
        w = [AV*y - λ*(V*y) for (λ,y) in zip(eig.values[1:k],eachcolumn(eig.vectors)[1:k])]
        V = [V w...]
        
        eigvalsN = eig.values[1:NTargetEigvals]
        display(eigvalsN)
        println("Norm(eigvalsN-eigvals):")
        println(norm(eigvalsN-eigvals))
        eigvals  = norm(eigvalsN-eigvals)>=ϑ ? eigvalsN : return eig.values
    end
    error("Davidson did not converge in $maxsubspacesize-dimensional subspace")
end
