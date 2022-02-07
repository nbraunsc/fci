#module diagonalize
#export diagonlize
using PyCall
using NPZ
using Printf
using Combinatorics
using LinearAlgebra
using StaticArrays
using Einsum
using TensorOperations
using JLD2
using fci

np = pyimport("numpy")

function run_fci(orbs, nalpha, nbeta, m=12)
    #get eigenvalues from lanczos
    int1e = npzread("/Users/nicole/code/fci/src/data/int1e_4.npy")
    int2e = npzread("/Users/nicole/code/fci/src/data/int2e_4.npy")
    H_pyscf = npzread("/Users/nicole/code/fci/src/data/H_full_a.npy")
    ci = npzread("/Users/nicole/code/fci/src/data/cimatrix.npy")
    yalpha, ybeta = make_xy(orbs, nalpha, nbeta)
    
    #get all configs
    #configa = zeros(UInt8, orbs)
    #configb = zeros(UInt8, orbs)
    #configa[1:nalpha] .= 1
    #configb[1:nbeta] .= 1
    
    configa = falses(orbs)
    configb = falses(orbs)
    configa[1:nalpha] .= true
    configb[1:nbeta] .= true

    alpha_configs = get_all_configs(configa, orbs, yalpha, nalpha)
    beta_configs = get_all_configs(configb, orbs, ybeta, nbeta)
    
    ndets_a = factorial(orbs)÷(factorial(nalpha)*factorial(orbs-nalpha))
    ndets_b = factorial(orbs)÷(factorial(nbeta)*factorial(orbs-nbeta))
    
    #make lookup table
    index_table_a, sign_table_a = make_index_table(alpha_configs, ndets_a, yalpha) 
    index_table_b, sign_table_b = make_index_table(beta_configs, ndets_b, ybeta) 
    
    Ha_diag = precompute_spin_diag_terms(alpha_configs, ndets_a, orbs, index_table_a, yalpha, int1e, int2e, nalpha)
    Hb_diag = precompute_spin_diag_terms(beta_configs, ndets_b, orbs, index_table_b, ybeta, int1e, int2e, nbeta)

    #get H components
    H_alpha = compute_ss_terms_full(ndets_a, orbs, int1e, int2e, index_table_a, alpha_configs, yalpha, sign_table_a)
    H_beta = compute_ss_terms_full(ndets_b, orbs, int1e, int2e, index_table_b, beta_configs, ybeta, sign_table_b)
    #H_mixed = compute_ab_terms_full(ndets_a, ndets_b, orbs, int1e, int2e, index_table_a, index_table_b, alpha_configs, beta_configs, yalpha, ybeta)
    
    H_alpha = Ha_diag + H_alpha
    H_beta = Hb_diag + H_beta

    Ia = Matrix{Float64}(I, size(H_alpha))
    Ib = Matrix{Float64}(I, size(H_beta))
    #Hmat = zeros(ndets_a*ndets_b, ndets_a*ndets_b)
    #Hmat .+= kron(Ib, H_alpha)
    #Hmat .+= kron(H_beta, Ia)
    #Hmat .+= H_mixed
    #H = .5*(Hmat+Hmat')
    #H = Hmat
    
    dim = ndets_a*ndets_b
    #b = rand(dim)
    b = zeros(dim)
    b[1] = 1
    #initalize empty matrices for T and V
    T = zeros(Float64, m+1, m)
    V = zeros(Float64, dim, m+1)
    #normalize start vector
    V[:,1] = b/norm(b)
    #next vector
    #w = Hmat*V[:,1]
    w = get_sigma(H_alpha, H_beta, Ia, Ib, V[:,1], [alpha_configs, beta_configs], orbs, [yalpha, ybeta], int2e, index_table_a, index_table_b, dim, ndets_a, sign_table_a, sign_table_b)
    #sigma, sigma3 = get_sigma(H_alpha, H_beta, Ia, Ib, V[:,1], [alpha_configs, beta_configs], orbs, [yalpha, ybeta], int2e, index_table_a, index_table_b, dim, ndets_a)
    #vector = V[:,1]
    #@save "/Users/nicole/code/fci/test/_testdata_h8.jld2" alpha_configs beta_configs yalpha ybeta index_table_a index_table_b Ha_diag H_alpha H_beta sigma3 sigma vector
    #orthogonalise
    T[1,1] = dot(w,V[:,1])
    w = w - T[1,1]*V[:,1]
    #normalize next vector
    T[2,1] = norm(w)
    V[:,2] = w/T[2,1]

    for j = 2:m
        #make T symmetric
        T[j-1, j] = T[j, j-1]
        
        #next vector
        w = get_sigma(H_alpha, H_beta, Ia, Ib, V[:,j], [alpha_configs, beta_configs], orbs, [yalpha, ybeta], int2e, index_table_a, index_table_b, dim, ndets_a, sign_table_a, sign_table_b)
        #w = Hmat*V[:,j]
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
        if T[j+1, j] < 10E-10
            @printf("\n\n --------------- Converged at %i iteration ---------------  \n\n", j)
            Tm = T[1:j, 1:j]
            #println("Tm tridiag matrix")
            #display(Tm)
            #Tm_vals = sort(eigvals(Tm))
            #println("\nTm eigen vals: ")
            #display(Tm_vals)
            #println("\nMy eigenvals - nuc:")
            #x = (my_energies .- nuc)
            #display(x)
            #diff = x[1]-Tm_vals[1]
            #println("\n -------- Diff in eigenvalues ------------")
            #display(diff)
            return Tm, V
            break
        end 

        if j == m
            println("\n ----------- HIT MAX ITERATIONS -------------\n ")
        end
    end
    
    #make T into symmetric matrix of shape (m,m)
    Tm = T[1:m, 1:m]
    return Tm, V
end

function old_make_index_table(configs, ndets, y_matrix)
    index_table = zeros(Int, configs[1].norbs, configs[1].norbs, ndets)#={{{=#
    orbs = [1:configs[1].norbs;]
    for I in 1:ndets
        vir = filter!(x->!(x in configs[I].config), [1:configs[I].norbs;])
        for p in configs[I].config
            for q in vir
                new_config, sorted_config, sign_s = old_excit_config(deepcopy(configs[I].config), [p,q])
                idx = get_index(new_config, y_matrix, configs[I].norbs)
                index_table[p,q,configs[I].index]=sign_s*idx
            end
        end
    end#=}}}=#
    return index_table#, sign_table
end

function make_index_table(configs, ndets, y_matrix)
    index_table = zeros(UInt, configs[1].norbs, configs[1].norbs, ndets)#={{{=#
    sign_table = zeros(configs[1].norbs, configs[1].norbs, ndets)
    orbs = [1:configs[1].norbs;]
    for I in 1:ndets
        vir = filter!(x->!(x in configs[I].config), [1:configs[I].norbs;])
        #idx_I = configs[I].index
        for p in configs[I].config
            for q in vir
                new_config, sorted_config, sign_s = excit_config(configs[I].config, p,q)
                idx = get_index(new_config, y_matrix, configs[I].norbs)
                sign_table[p,q,configs[I].index]=sign_s
                index_table[p,q,configs[I].index]=idx
            end
        end
    end#=}}}=#
    return index_table, sign_table
end

function precompute_spin_diag_terms(configs, ndets, orbs, index_table, y_matrix, int1e, int2e, nelec)
    Hout = zeros(ndets, ndets)#={{{=#
    for K in 1:ndets
        config = configs[K].config
        idx = configs[K].index
        #idx = get_index(config, y_matrix, orbs)
        for i in 1:nelec
            Hout[idx,idx] += int1e[config[i], config[i]]
            for j in i+1:nelec
                Hout[idx,idx] += int2e[config[i], config[i], config[j], config[j]]
                Hout[idx,idx] -= int2e[config[i], config[j], config[i], config[j]]
            end
        end
    end#=}}}=#
    return Hout
end

function old_compute_ss_terms_full(ndets, norbs, int1e, int2e, index_table, configs, y_matrix)
    Ha = zeros(ndets, ndets)#={{{=#
    nelecs = size(configs[1].config)[1]
    
    #h1eff = deepcopy(int1e)
    #@tensor begin
    #    h1eff[p,q] -= .5 * int2e[p,j,j,q]
    #end
    
    for I in configs #Ia or Ib, configs=list of all possible determinants
        #I_idx = get_index(I.config, y_matrix, I.norbs)
        I_idx = I.index
        F = zeros(ndets)
        orbs = [1:I.norbs;]
        vir = filter!(x->!(x in I.config), orbs)
       
        #single excitation
        for k in I.config
            for l in vir
                #annihlate electron in orb k
                config_single, sorted_config, sign_s = old_excit_config(deepcopy(I.config), [k,l])
                config_single_idx = index_table[k,l,I_idx]
                #config_single_idx = index_table[k,l,I.label]
                F[abs(config_single_idx)] += sign_s*int1e[k,l]
                for m in I.config
                    if m!=k
                        F[abs(config_single_idx)] += sign_s*(int2e[k,l,m,m]-int2e[k,m,l,m])
                    end
                end
            end
        end
        
        #double excitation
        for k in I.config
            for i in I.config
                if i>k
                    for l in vir
                        for j in vir
                            if j>l
                                single, sorted_s, sign_s = old_excit_config(deepcopy(I.config), [k,l])
                                double, sorted_d, sign_d = old_excit_config(deepcopy(sorted_s), [i,j])
                                idx = get_index(double, y_matrix, I.norbs)
                                if sign_d == sign_s
                                    F[abs(idx)] += (int2e[i,j,k,l] - int2e[i,l,j,k]) #one that works for closed

                                else
                                    F[abs(idx)] -= (int2e[i,j,k,l] - int2e[i,l,j,k]) #one that works for closed
                                end
                            end
                        end
                    end
                end
            end
        end

            
        Ha[:,I_idx] .= F
    end#=}}}=#
    return Ha
end

function compute_ss_terms_full(ndets, norbs, int1e, int2e, index_table, configs, y_matrix, sign_table)
    Ha = zeros(ndets, ndets)#={{{=#
    nelecs = size(configs[1].config)[1]
    
    #h1eff = deepcopy(int1e)
    #@tensor begin
    #    h1eff[p,q] -= .5 * int2e[p,j,j,q]
    #end
    
    for I in configs #Ia or Ib, configs=list of all possible determinants
        I_idx = I.index
        F = zeros(ndets)
        orbs = [1:I.norbs;]
        vir = filter!(x->!(x in I.config), orbs)
       
        #single excitation
        for k in I.config
            for l in vir
                #annihlate electron in orb k
                config_single_idx = index_table[k,l,I_idx]
                sign_s = sign_table[k,l,I_idx]
                F[abs(config_single_idx)] += sign_s*int1e[k,l]
                for m in I.config
                    if m!=k
                        F[abs(config_single_idx)] += sign_s*(int2e[k,l,m,m]-int2e[k,m,l,m])
                    end
                end
            end
        end
        
        #double excitation
        for k in I.config
            for i in I.config
                if i>k
                    for l in vir
                        for j in vir
                            if j>l
                                single, sorted_s, sign_s = excit_config(I.config, k,l)
                                double, sorted_d, sign_d = excit_config(sorted_s, i,j)
                                idx = get_index(double, y_matrix, I.norbs)
                                if sign_d == sign_s
                                    F[abs(idx)] += (int2e[i,j,k,l] - int2e[i,l,j,k]) #one that works for closed

                                else
                                    F[abs(idx)] -= (int2e[i,j,k,l] - int2e[i,l,j,k]) #one that works for closed
                                end
                            end
                        end
                    end
                end
            end
        end
        ##single excitation
        #for k in I.config
        #    for l in vir
        #        #annihlate electron in orb k
        #        config_single_idx = index_table[k,l,I_idx]
        #        sign_s = sign_table[k,l,I_idx]
        #        F[abs(config_single_idx)] += sign_s*int1e[k,l]
        #        for m in I.config
        #            if m!=k
        #                F[abs(config_single_idx)] += sign_s*(int2e[k,l,m,m]-int2e[k,m,l,m])
        #            end
        #        end
        #    end
        #end
        #
        ##double excitation
        #for k in I.config
        #    for i in I.config
        #        if i>k
        #            for l in vir
        #                for j in vir
        #                    if j>l
        #                        single, sorted_s, sign_s = excit_config(I.config, k,l)
        #                        double, sorted_d, sign_d = excit_config(sorted_s, i,j)
        #                        idx = get_index(double, y_matrix, I.norbs)
        #                        if sign_d == sign_s
        #                            F[abs(idx)] += (int2e[i,j,k,l] - int2e[i,l,j,k]) #one that works for closed

        #                        else
        #                            F[abs(idx)] -= (int2e[i,j,k,l] - int2e[i,l,j,k]) #one that works for closed
        #                        end
        #                    end
        #                end
        #            end
        #        end
        #    end
        #end

            
        Ha[:,I_idx] .= F
    end#=}}}=#
    return Ha
end
        
function compute_ab_terms_full(ndets_a, ndets_b, norbs, int1e, int2e, index_table_a, index_table_b, alpha, beta, yalpha, ybeta)
    dim = ndets_a*ndets_b#={{{=#
    Hmat = zeros(dim, dim)

    for Ka in alpha
        Ka_idx = get_index(Ka.config, yalpha, norbs)
        orbsa = [1:norbs;]
        vira = filter!(x->!(x in Ka.config), orbsa)
        for Kb in beta
            Kb_idx = get_index(Kb.config, ybeta, norbs)
            orbsb = [1:norbs;]
            virb = filter!(x->!(x in Kb.config), orbsb)
            K = Ka_idx + (Kb_idx-1)*ndets_a #works for closed shell
            
            #diagonal part
            for l in Kb.config
                for n in Ka.config
                    Hmat[K, K] += int2e[n,n,l,l]
                end
            end
            
            #excit alpha only
            for p in Ka.config
                for q in vira
                    a_single, sort_a, sign_a = excit_config(deepcopy(Ka.config), [p,q])
                    idxa = abs(index_table_a[p,q,Ka.label])

                    Kprime = idxa + (Kb_idx-1)*ndets_a
                    #alpha beta <ii|jj>
                    for m in Kb.config
                        Hmat[K,Kprime]+=sign_a*int2e[p,q,m,m]
                    end
                end
            end

            #excit beta only
            for r in Kb.config
                for s in virb
                    b_single, sort_b, sign_b = excit_config(deepcopy(Kb.config), [r,s])
                    idxb = abs(index_table_b[r,s,Kb.label])
                    Lprime = Ka_idx + (idxb-1)*ndets_a
                    
                    #alpha beta <ii|jj>
                    for n in Ka.config
                        Hmat[K,Lprime]+=sign_b*int2e[r,s,n,n]
                    end
                end
            end

            #excit alpha and beta
            for p in Ka.config
                for q in vira
                    a_single, sort_a, sign_a = excit_config(deepcopy(Ka.config), [p,q])
                    idxa = abs(index_table_a[p,q,Ka.label])
                    for r in Kb.config
                        for s in virb
                            b_single, sort_b, sign_b = excit_config(deepcopy(Kb.config), [r,s])
                            idxb = abs(index_table_b[r,s,Kb.label])
                            L = idxa + (idxb-1)*ndets_a
                            Hmat[K,L] += sign_a*sign_b*(int2e[p,q,r,s])
                        end
                    end
                end
            end
        end
    end#=}}}=#
    return Hmat
end

function old_get_sigma3(configs, norbs, y_matrix, int2e, vector, index_table_a, index_table_b, dim, ndets_a)
    #configs = [alpha_configs, beta_configs]{{{
    #nelec = [n_alpha, n_beta]
    #y_matrix = [y_alpha, y_beta]
    sigma3 = zeros(dim, 1)

    for I in configs[1]  #alpha configs
        orbs = [1:norbs;]
        vir_I = filter!(x->!(x in I.config), orbs)
        #I_idx = get_index(I.config, y_matrix[1], norbs)
        I_idx = I.index
        for J in configs[2]     #beta configs
            orbs2 = [1:norbs;]
            vir_J = filter!(x->!(x in J.config), orbs2)
            #J_idx = get_index(J.config, y_matrix[2], norbs)
            J_idx = J.index
            K = I_idx + (J_idx-1)*ndets_a

            for l in J.config
                for n in I.config
                    #Hmat[K,K]
                    sigma3[K] += int2e[n,n,l,l]*vector[K]
                end
            end

            #excit alpha only
            for p in I.config
                for q in vir_I
                    a_single, sorta, sign_a = old_excit_config(deepcopy(I.config), [p,q])
                    idxa = index_table_a[p,q,I_idx]
                    Kprim = idxa + (J_idx-1)*ndets_a
                    for m in J.config
                        #Hmat[K,Kprim]
                        sigma3[K]+=sign_a*int2e[p,q,m,m]*vector[Kprim]
                    end
                end
            end

            #excit beta only
            for r in J.config
                for s in vir_J
                    b_single, sortb, sign_b = old_excit_config(deepcopy(J.config), [r,s])
                    idxb = index_table_b[r,s,J_idx]
                    Lprim = I_idx + (idxb-1)*ndets_a
                    for n in I.config
                        #Hmat[K,Lprim]
                        sigma3[K]+=sign_b*int2e[r,s,n,n]*vector[Lprim]
                    end
                end
            end

            #excit both alpha and beta
            for p in I.config
                for q in vir_I
                    a_single, sorta, sign_a = old_excit_config(deepcopy(I.config), [p,q])
                    idxa = index_table_a[p,q,I_idx]
                    for r in J.config
                        for s in vir_J
                            b_single, sortb, sign_b = old_excit_config(deepcopy(J.config), [r,s])
                            idxb = index_table_b[r,s,J_idx]
                            L = idxa + (idxb-1)*ndets_a
                            #Hmat[K,L]
                            sigma3[K] += sign_a*sign_b*int2e[p,q,r,s]*vector[L]
                        end
                    end
                end
            end
        end
    end#=}}}=#
    return sigma3
end

function get_sigma3(configs, norbs, int2e, vector, index_table_a, index_table_b, dim, ndets_a, sign_table_a, sign_table_b)
    #configs = [alpha_configs, beta_configs]{{{
    #nelec = [n_alpha, n_beta]
    
    #sigma3 = MVector{dim, Int}
    sigma3 = zeros(dim, 1) #make this to MVector, but can't figure out indexing

    for I in configs[1]  #alpha configs
        orbs = [1:norbs;] ## change this to SVector
        vir_I = filter!(x->!(x in I.config), orbs)
        I_idx = I.index
        for J in configs[2]     #beta configs
            orbs2 = [1:norbs;]
            vir_J = filter!(x->!(x in J.config), orbs2)
            J_idx = J.index
            K = I_idx + (J_idx-1)*ndets_a

            for l in J.config
                for n in I.config
                    #Hmat[K,K]
                    sigma3[K] += int2e[n,n,l,l]*vector[K]
                end
            end

            #excit alpha only
            for p in I.config
                for q in vir_I
                    idxa = index_table_a[p,q,I_idx]
                    sign_a = sign_table_a[p,q,I_idx]
                    Kprim = idxa + (J_idx-1)*ndets_a
                    for m in J.config
                        #Hmat[K,Kprim]
                        sigma3[K]+=sign_a*int2e[p,q,m,m]*vector[Kprim]
                    end
                end
            end

            #excit beta only
            for r in J.config
                for s in vir_J
                    idxb = index_table_b[r,s,J_idx]
                    sign_b = sign_table_b[r,s,J_idx]
                    Lprim = I_idx + (idxb-1)*ndets_a
                    for n in I.config
                        #Hmat[K,Lprim]
                        sigma3[K]+=sign_b*int2e[r,s,n,n]*vector[Lprim]
                    end
                end
            end

            #excit both alpha and beta
            for p in I.config
                for q in vir_I
                    idxa = index_table_a[p,q,I_idx]
                    sign_a = sign_table_a[p,q,I_idx]
                    for r in J.config
                        for s in vir_J
                            idxb = index_table_b[r,s,J_idx]
                            sign_b = sign_table_b[r,s,J_idx]
                            L = idxa + (idxb-1)*ndets_a
                            #Hmat[K,L]
                            sigma3[K] += sign_a*sign_b*int2e[p,q,r,s]*vector[L]
                        end
                    end
                end
            end
        end
    end#=}}}=#
    return sigma3
end

function get_sigma(Ha, Hb, Ia, Ib, vector, configs, norbs, y_matrix, int2e, index_table_a, index_table_b, dim, ndets_a, sign_table_a, sign_table_b)
    #b = reshape(vector, (size(Ha)[1], size(Hb)[1]))
    #sigma1 = reshape(Ha*b, (dim, 1))
    #sigma2 = reshape(Hb*transpose(b), (dim, 1))
    sigma1 = kron(Ib, Ha)*vector
    sigma2 = kron(Hb, Ia)*vector
    sigma3 = get_sigma3(configs, norbs, int2e, vector, index_table_a, index_table_b, dim, ndets_a, sign_table_a, sign_table_b)
    sigma = sigma1 + sigma2 + sigma3
    return sigma
end

function make_xy(norbs, nalpha, nbeta)
    #makes y matrices for grms indexing{{{
    n_unocc_a = (norbs-nalpha)+1
    n_unocc_b = (norbs-nbeta)+1

    #make x matricies
    xalpha = zeros(n_unocc_a, nalpha+1)
    xbeta = zeros(n_unocc_b, nbeta+1)
    #fill first row and columns
    xalpha[:,1] .= 1
    xbeta[:,1] .= 1
    xalpha[1,:] .= 1
    xbeta[1,:] .= 1
    
    for i in 2:nalpha+1
        for j in 2:n_unocc_a
            xalpha[j, i] = xalpha[j-1, i] + xalpha[j, i-1]
        end
    end

    for i in 2:nbeta+1
        for j in 2:n_unocc_b
            xbeta[j, i] = xbeta[j-1, i] + xbeta[j, i-1]
        end
    end

    #make y matrices
    copya = deepcopy(xalpha)
    copyb = deepcopy(xbeta)
    arraya = zeros(size(xalpha)[2])
    arrayb = zeros(size(xbeta)[2])
    yalpha = vcat(transpose(arraya), xalpha[1:size(xalpha)[1]-1, :])
    ybeta = vcat(transpose(arrayb), xbeta[1:size(xbeta)[1]-1, :])#=}}}=#
    return yalpha, ybeta
end
function old_get_all_configs(config, norbs)
    #get all possible configs from a given start config{{{
    nelecs = UInt8(size(config)[1])
    all_configs = unique(collect(permutations(config, size(config)[1])))
    configs = [] #empty Array
    for i in 1:size(all_configs)[1]
        A = findall(!iszero, all_configs[i])
        push!(configs,fci.DeterminantString(norbs, nelecs, A, 1, N=nelecs)) 
    end
    #=}}}=#
    return configs
end

function get_all_configs(config, norbs, y, nelecs)
    #get all possible configs from a given start config{{{
    all_configs = unique(collect(permutations(config, size(config)[1])))
    configs = [] #empty Array
    for i in 1:size(all_configs)[1]
        A = findall(all_configs[i])
        b = SVector{nelecs}(A)
        idx = get_index(A, y, norbs)
        push!(configs,fci.DeterminantString(norbs, nelecs, b, idx)) 
    end
    #=}}}=#
    return configs
end

function old_get_index(config, y, norbs)
    #config has to be orbital index form since it will turned into bit string form{{{
    string = zeros(UInt, norbs)
    string[config] .= 1

    #println("config: ", config)
    index = 1
    start = [1,1]

    for value in string
        if value == 0
            #move down but dont add to index
            start[1] = start[1]+1
        end
        if value == 1
            #move right and add value to index
            start[2] = start[2] + 1
            index += y[start[1], start[2]]
        end
    end#=}}}=#
    return UInt8(index)
end

function get_index(config, y, norbs)
    #config has to be orbital index form since it will turned into bit string form{{{
    string = falses(norbs)
    string[config] .= true

    index = 1
    start = [1,1]

    for value in string
        if value
            #move right and add value to index
            start[2] = start[2] + 1
            index += y[start[1], start[2]]
        else
            #move down but dont add to index
            start[1] = start[1]+1
        end
    end#=}}}=#
    return Int(index)
end

function old_excit_config(config, positions)
    #apply creation operator to the string{{{
    #get new index of string and store in lookup table
    #config is orbital indexing in ascending order
    #positions [i,a] meaning electron in orb_i went to orb_a
    
    config = Vector(config)
    spot = first(findall(x->x==positions[1], config))
    config[spot] = positions[2]
    config_org = deepcopy(config) 
    count, arr = bubble_sort(config)
    if iseven(count)
        sign = 1
    else
        sign = -1
    end#=}}}=#
    return config_org, arr, sign
end

function excit_config(config, i, j)
    #apply creation operator to the string{{{
    #get new index of string and store in lookup table
    #config is orbital indexing in ascending order
    #positions [i,a] meaning electron in orb_i went to orb_a
    #config = SVector
    
    spot = first(findall(x->x==i, config))
    new = Vector(config)
    new[spot] = j
    count, arr = bubble_sort(new)
    if iseven(count)
        sign = 1
    else
        sign = -1
    end#=}}}=#
    return new, arr, sign
end

function bubble_sort(arr)
    len = size(arr)[1]#={{{=#
    count = 0
    # Traverse through all array elements
    for i = 1:len-1
        for j = 2:len
        # Last i elements are already in place
        # Swap if the element found is greater
            if arr[j-1] > arr[j] 
                count += 1
                tmp = arr[j-1]
                arr[j-1] = arr[j]
                arr[j] = tmp
            end
        end
    end
    return count, arr#=}}}=#
end

