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

np = pyimport("numpy")

h = npzread("/Users/nicole/code/fci/h1.npy")
h4 = npzread("/Users/nicole/code/fci/h4.npy")
h5 = npzread("/Users/nicole/code/fci/h5.npy")

display(h)
display(h4)
add = h4+h
diff = h5-add
display(diff)
y=z
mutable struct DeterminantString
    norbs::UInt8
    nelec::UInt8
    config::Vector{Int}
    #config::MVector{N, Int}
    index::UInt
    label::UInt
end

#orbs = 5
#nalpha = 3
#nbeta = 2
function run_fci(orbs, nalpha, nbeta, m=12)
    #get eigenvalues from lanczos{{{
    int1e = npzread("/Users/nicole/code/fci/src/data/int1e_4.npy")
    int2e = npzread("/Users/nicole/code/fci/src/data/int2e_4.npy")
    #int2e = zeros(size(int2e1))
    H_pyscf = npzread("/Users/nicole/code/fci/src/data/H_full_a.npy")
    ci = npzread("/Users/nicole/code/fci/src/data/cimatrix.npy")
    yalpha, ybeta = make_xy(orbs, nalpha, nbeta)
    
    #get all configs
    configa = zeros(orbs)
    configb = zeros(orbs)
    configa[1:nalpha] .= 1
    configb[1:nbeta] .= 1


    alpha_configs = get_all_configs(configa, orbs)
    beta_configs = get_all_configs(configb, orbs)
    ndets_a = factorial(orbs)÷(factorial(nalpha)*factorial(orbs-nalpha))
    ndets_b = factorial(orbs)÷(factorial(nbeta)*factorial(orbs-nbeta))
    
    #make lookup table
    index_table_a = make_index_table(alpha_configs, ndets_a, yalpha) 
    index_table_b = make_index_table(beta_configs, ndets_b, ybeta) 
    
    Ha_diag = precompute_spin_diag_terms(alpha_configs, ndets_a, orbs, index_table_a, yalpha, int1e, int2e, nalpha)
    Hb_diag = precompute_spin_diag_terms(beta_configs, ndets_b, orbs, index_table_b, ybeta, int1e, int2e, nbeta)

    #get H components
    H_alpha = compute_ss_terms_full(ndets_a, orbs, int1e, int2e, index_table_a, alpha_configs, yalpha)
    H_beta = compute_ss_terms_full(ndets_b, orbs, int1e, int2e, index_table_b, beta_configs, ybeta)
    H_mixed = compute_ab_terms_full(ndets_a, ndets_b, orbs, int1e, int2e, index_table_a, index_table_b, alpha_configs, beta_configs, yalpha, ybeta)
    
    H_alpha = Ha_diag + H_alpha
    H_beta = Hb_diag + H_beta
    #H_alpha = Ha_diag
    #H_beta = Hb_diag

    H_alpha[abs.(H_alpha) .< 1e-14] .= 0
    H_beta[abs.(H_beta) .< 1e-14] .= 0
    H_mixed[abs.(H_mixed) .< 1e-14] .= 0
    H_pyscf[abs.(H_pyscf) .< 1e-14] .= 0
    ci[abs.(ci) .< 1e-14] .= 0


    Ia = Matrix{Float64}(I, size(H_alpha))
    Ib = Matrix{Float64}(I, size(H_beta))
    Ha = kron(H_alpha, Ib) 
    Hb = kron(Ia, H_beta) 
    Ha[abs.(Ha) .< 1e-14] .= 0
    Hb[abs.(Hb) .< 1e-14] .= 0

    Hmat = zeros(ndets_a*ndets_b, ndets_a*ndets_b)
    Hmat .+= kron(Ib, H_alpha)
    Hmat .+= kron(H_beta, Ia)
    Hmat .+= H_mixed
    
    #H = .5*(Hmat+Hmat')
    H = Hmat
    npzwrite("h1.npy", H)
    
    #H = kron(H_alpha, Ib) + kron(Ia, H_beta) + H_mixed 

    println("\nH_alpha")
    display(H_alpha)
    
    println("\nH_beta")
    display(H_beta)

    println("H mixed")
    display(H_mixed)
    display(Ha)
    display(Hb)
    
    full_a = kron(Ib, H_alpha)
    full_a[abs.(full_a) .< 1e-14] .=0
    println("H alpha full")
    display(full_a)
     
    full_b = kron(H_beta, Ia)
    full_b[abs.(full_b) .< 1e-14] .=0
    println("H beta full")
    display(full_b)
    
    println("\nmy H:")
    display(H)
    
    println("my ci:")
    display(ci)

    println("\n Diff between my ci and slater ci")
    diff3 = ci-H
    diff3[abs.(diff3) .< 1e-14] .= 0
    display(diff3)
    println(diag(diff3))

    e = eigvals(H)
    e2 = eigvals(ci)
    e3 = eigvals(H_pyscf)
    println(e)
    println(e2)
    println(e3)
    println(e-e2)
    println("\n")
    println(e-e3)
    y=z

    println("\nH Pyscf")
    display(H_pyscf)
    
    println("\nDiff Between PYSCF and MY H")
    diff = H_pyscf - H
    diff[abs.(diff) .< 1e-14] .= 0
    display(diff)
    println("Trace: ", tr(diff))
    println(diag(diff))

    println("pyscf-ha-hb")
    diffab = H_pyscf-kron(Ia, H_beta)-kron(H_alpha,Ib)
    #diffab = H_pyscf-kron(Ib, H_alpha)-kron(H_beta,Ia)
    diffab[abs.(diffab) .< 1e-14] .= 0
    display(diffab)
    display(H_mixed)

    #println(H_pyscf[13,1], " ", H_pyscf[19,1])
    #println(H[13,1], " ", H[19,1])
    #println(diff[13,1], " ", diff[19,1])}}}
    
    #={{{=#

    #b = zeros(size(H_mixed)[1])
    #b[1] = 1
    #b = rand(Float64, size(H_mixed)[1])
    #initalize empty matrices for T and V
    T = zeros(Float64, m+1, m)
    V = zeros(Float64, size(H_mixed)[1], m+1)
    #normalize start vector
    V[:,1] = b/norm(b)
    #next vector
    w = get_sigma(H_alpha, H_beta, V[:,1])

    #w = matrix*V[:,1]
    #orthogonalise
    T[1,1] = dot(w,V[:,1])
    w = w - T[1,1]*V[:,1]
    #normalze next vector
    T[2,1] = norm(w)
    V[:,2] = w/T[2,1]
    print("B values: trying \n")
    for j = 2:m
        #make T symmetric
        println(T[j, j-1])
        
        T[j-1, j] = T[j, j-1]
        
        #next vector
        w = get_sigma(H_alpha, H_beta, H_mixed, V[:,j])
        #w = matrix*V[:,j]
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
    return Tm, V#=}}}=#
end

function make_index_table(configs, ndets, y_matrix)
    index_table = zeros(Int, configs[1].norbs, configs[1].norbs, ndets)#={{{=#
    #sign_table = trues(configs[1].norbs, configs[1].norbs, ndets)
    orbs = [1:configs[1].norbs;]
    for I in 1:ndets
        vir = filter!(x->!(x in configs[I].config), [1:configs[I].norbs;])
        for p in configs[I].config
            for q in vir
                new_config, sorted_config, sign_s = excit_config(deepcopy(configs[I].config), [p,q])
                #println(new_config, typeof(new_config))
                idx = get_index(new_config, y_matrix, configs[I].norbs)
                index_table[p,q,configs[I].label]=sign_s*idx
                #index_table[p,q,I]=sign_s*idx
                #if sign_s < 0
                #    sign_table[p,q,I]=false
                #end
            end
        end
    end#=}}}=#
    return index_table#, sign_table
end

function precompute_spin_diag_terms(configs, ndets, orbs, index_table, y_matrix, int1e, int2e, nelec)
    Hout = zeros(ndets, ndets)
    for K in 1:ndets
        config = configs[K].config
        idx = get_index(config, y_matrix, orbs)
        for i in 1:nelec
            Hout[idx,idx] += int1e[config[i], config[i]]
            for j in i+1:nelec
                Hout[idx,idx] += int2e[config[i], config[i], config[j], config[j]]
                Hout[idx,idx] -= int2e[config[i], config[j], config[i], config[j]]
            end
        end
    end
    return Hout
end

function compute_ss_terms_full(ndets, norbs, int1e, int2e, index_table, configs, y_matrix)
    Ha = zeros(ndets, ndets)
    nelecs = size(configs[1].config)[1]
    
    h1eff = deepcopy(int1e)
    @tensor begin
        h1eff[p,q] -= .5 * int2e[p,j,j,q]
    end
    
    for I in configs #Ia or Ib, configs=list of all possible determinants
        I_idx = get_index(I.config, y_matrix, I.norbs)
        F = zeros(ndets)
        orbs = [1:I.norbs;]
        vir = filter!(x->!(x in I.config), orbs)
        #println("\n I", I_idx)
        #println(I.config)
       
        #single excitation
        for k in I.config
            for l in vir
                #annihlate electron in orb k
                config_single, sorted_config, sign_s = excit_config(deepcopy(I.config), [k,l])
                config_single_idx = index_table[k,l,I.label]
                #@inbounds F[abs(config_single_idx)] += sign_s*h1eff[k,l] #h_kl prime
                #println("adding single to spot: ", config_single_idx)
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
                                single, sorted_s, sign_s = excit_config(deepcopy(I.config), [k,l])
                                double, sorted_d, sign_d = excit_config(deepcopy(sorted_s), [i,j])
                                idx = get_index(double, y_matrix, I.norbs)
                                if sign_d == sign_s
                                    #F[abs(idx)] += (int2e[i,j,k,l] - int2e[i,k,j,l])
                                    F[abs(idx)] += (int2e[i,j,k,l] - int2e[i,l,j,k]) #one that works for closed

                                else
                                    #F[abs(idx)] -= (int2e[i,j,k,l] - int2e[i,k,j,l])
                                    F[abs(idx)] -= (int2e[i,j,k,l] - int2e[i,l,j,k]) #one that works for closed
                                end
                            end
                        end
                    end
                end
            end
        end

            
        #        for i in config_single
        #            if i != l
        #                for j in vir  #annihlate electron (from single excited config) in orb i != l (previouslly excited electron)
        #                    if j!=l
        #                        config_double, sorted_double, sign_d = excit_config(deepcopy(sorted_config), [i, j])
        #                        config_double_idx = get_index(config_double, y_matrix, I.norbs)
        #                        println("next j iteration")
        #                        println(config_double, "idx: ", config_double_idx)
        #                        if sign_d == sign_s
        #                            println("same sign")
        #                            println("adding to spot : ", config_double_idx)
        #                            @inbounds F[abs(config_double_idx)] += .5 * int2e[i,k,j,l]
        #                            #@inbounds F[config_double_idx] += .5 * int2e[i,j,k,l]
        #                        else
        #                            println("not same sign")
        #                            println("adding to spot : ", config_double_idx)
        #                            @inbounds F[abs(config_double_idx)] -= .5 * int2e[i,k,j,l]
        #                            #@inbounds F[config_double_idx] -= .5 * int2e[i,j,k,l]
        #                        end
        #                    end
        #                end
        #            end
        #        end
        #    end
        #end
        #display(F)
        Ha[:,I_idx] .= F
        #display(Ha)
    end#=}}}=#
    return Ha
end
      

function old_compute_ss_terms(ndets, norbs, int1e, int2e, index_table, configs, y_matrix)
    #something here adds to the diagonal when it shouldnt!!!{{{
    Ha = zeros(ndets, ndets)
    
    h1eff = deepcopy(int1e)
    @tensor begin
        h1eff[p,q] -= .5 * int2e[p,j,j,q]
    end

    F = zeros(ndets)
    display(h1eff)

    for I in 1:ndets
        idx = get_index(configs[I].config, y_matrix, norbs)
        F .= 0
        println("\n", I)
        println(idx)
        println(configs[I].config)

        for k in 1:norbs
            if k ∉ configs[I].config #not in
                println(configs[I].config)
                println("k not in config: ", k)
                continue
            end

            for l in 1:norbs
                if l in configs[I].config
                    println("already in config, cant excit")
                    continue
                end

                K = index_table[k,l,idx]
                excit, sorted, signs = excit_config(deepcopy(configs[I].config), [k,l])
                println("excited config: ", excit)
                println("k->l ", k, l)
                println("new index: ", K)

                if K == 0
                    println("no excitation possible")
                    continue
                end
                
                sign_kl = sign(K)
                K=abs(K)
                @inbounds F[K] += sign_kl*h1eff[k,l] #h_kl prime
                display(F)
                
                for i in 1:norbs
                    if i ∉ excit #not in
                        println("i not in single excit: ", i, " ", excit)
                        continue
                    end

                    for j in 1:norbs
                        if l in excit || l in configs[I].config
                            println("already in excit")
                            continue
                        end

                        J=index_table[i,j,K]
                        println("K: ", K)
                        println("get_index: ", get_index(excit, y_matrix, norbs))
                        println("single excit: ", excit)
                        println("i->j ", i, j)
                        println("J: ", J)
                        
                        if J == 0
                            println("no double excit possible")
                            continue
                        end
                        
                        println(i,j)
                        #println("double excited config: ", excit2)
                        println("i->j ", i, j)
                        println("new index: ", J)
                        
                        sign_ij = sign(J)
                        J=abs(J)
                        
                        if sign_kl == sign_ij
                            @inbounds F[J] += .5 * int2e[i,j,k,l]
                            display(F)
                        else
                            @inbounds F[J] -= .5 * int2e[i,j,k,l]
                            display(F)
                        end
                    end
                end
            end
        end
        Ha[:,idx] .= F
    end
    return Ha#=}}}=#
end
        
function compute_ab_terms_full(ndets_a, ndets_b, norbs, int1e, int2e, index_table_a, index_table_b, alpha, beta, yalpha, ybeta)
    dim = ndets_a*ndets_b
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
                    idx_mine = get_index(sort_a, yalpha, norbs)
                    if idxa!= idx_mine
                        println("idexes dont match")
                        y=z
                    end
                    if idxa == K
                        println("diagonal term")
                        y=z
                    end


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
                    idx_mine = get_index(sort_b, ybeta, norbs)
                    if idxb!= idx_mine
                        println("idexes dont match")
                        y=z
                    end
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
    end
    return Hmat
end


    
    #for Kb in 1:ndets_b 
    #    idxb = get_index(beta[Kb].config, ybeta, norbs)
    #    for Ka in 1:ndets_a
    #        idxa = get_index(alpha[Ka].config, yalpha, norbs)
    #        K = idxa + (idxb-1)*ndets_a

    #        #Diagonal part of mixed term!
    #        for n in alpha[Ka].config
    #            for l in beta[Kb].config
    #                Hmat[K,K] += int2e[n,n,l,l]
    #            end
    #        end

    #        for r in 1:norbs
    #            for p in 1:norbs
    #                La = index_table_a[p,r,idxa]
    #                #La = index_table_a[p,r,Ka]
    #                if La == 0
    #                    continue
    #                end
    #                #config_single, sorted_config, sign_s = excit_config(deepcopy(alpha[Ka].config), [p,r])
    #                #println("alpha excit: ", config_single, " sign: ", sign_s)
    #                #println("p->r: ", p,r)
    #                sign_a = sign(La)
    #                #println("sign from table: ", sign_a)
    #                La = abs(La)
    #                Lb=1
    #                sign_b =1
    #                L=1
    #                for s in 1:norbs
    #                    for q in 1:norbs
    #                        Lb = index_table_b[q,s,idxb]
    #                        if Lb == 0
    #                            continue
    #                        end
    #                        sign_b = sign(Lb)
    #                        Lb = abs(Lb)
    #                        L = La + (Lb-1)*ndets_a
    #                        Hmat[K,L] += sign_a*sign_b*(int2e[p,q,r,s] - int2e[p,r,q,s])
    #                        #Hmat[K,L] += sign_a*sign_b*(int2e[p,r,q,s] -int2e[p,q,r,s])
    #                        #Hmat[K,L] += int2e[p,r,q,s]*sign_a*sign_b
    #                    end
    #                end
    #            end
    #        end
    #    end
    #end#=}}}=#

function get_sigma3(configs, nelec, norbs, y_matrix, int2e, vector)
    #configs = [alpha_configs, beta_configs]{{{
    #nelec = [n_alpha, n_beta]
    #y_matrix = [y_alpha, y_beta]
    ndets_a = factorial(norbs) / (factorial(nelec[1])*factorial(norbs-nelec[1])) 
    ndets_b = factorial(norbs) / (factorial(nelec[2])*factorial(norbs-nelec[2])) 
    size_mixed = Int(ndets_a*ndets_b)
    sigma3 = zeros(size_mixed, 1)

    for I in configs[1]  #alpha configs
        orbs = [1:norbs;]
        vir_I = filter!(x->!(x in I), orbs)
        for i in I
            for a in vir_I
                for J in configs[2]     #beta configs
                    orbs2 = [1:norbs;]
                    vir_J = filter!(x->!(x in J), orbs2)
                    for j in J
                        for b in vir_J
                            #i think ill have some double counting in these
                            I_prim, signi = excit_config(deepcopy(I), [i, a])
                            J_prim, signj = excit_config(deepcopy(J), [j, b])
                            I_prim_idx = get_index(I_prim, y_matrix[1], norbs)
                            J_prim_idx = get_index(J_prim, y_matrix[2], norbs)
                            #println("I prim: ", I_prim, "index: ", I_prim_idx)
                            #println("J prim: ", J_prim, "index: ", J_prim_idx)
                            #Hmixed[I_prim_idx, J_prim_idx] += signi*signj*int2e[i, a, j, b]
                            #Hmixed[J_prim_idx, I_prim_idx] += signi*signj*int2e[i, a, j, b]
                            sigma3[I_prim_idx] += signi*signj*int2e[i, j, a, b]*vector[J_prim_idx]
                        end
                    end
                end
            end
        end
    end#=}}}=#
    return sigma3
end

function get_sigma(Ha, Hb, vector)
    b = reshape(vector, (size(Ha)[1], size(Hb)[1]))
    sigma1 = reshape(-Ha*b, (size(Hmixed)[1], 1))
    sigma2 = reshape(-Hb*transpose(b), (size(Hmixed)[1], 1))
    sigma3 = get_sigma3(configs, nelec, norbs, y_matrix, int2e, vector)
    #sigma3 = -Hmixed*vector
    sigma = sigma1 + sigma2 + sigma3
    return sigma
end

function make_xy(norbs, nalpha, nbeta)
    #makes y mazotrices for grms indexing{{{
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

function get_all_configs(config, norbs)
    #get all possible configs from a given start config{{{
    nelecs = UInt8(size(config)[1])
    all_configs = unique(collect(permutations(config, size(config)[1])))
    configs = [] #empty Array
    for i in 1:size(all_configs)[1]
        A = findall(!iszero, all_configs[i])
        push!(configs,DeterminantString(norbs, nelecs, A, 1, UInt(i))) 
    end
    #=}}}=#
    return configs
end

function get_index(config, y, norbs)
    #config has to be orbital index form since it will turned into bit string form{{{
    string = zeros(Int8, norbs)
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

function excit_config(config, positions)
    #apply creation operator to the string{{{
    #get new index of string and store in lookup table
    #config is orbital indexing in ascending order
    #positions [i,a] meaning electron in orb_i went to orb_a
    
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

function one_elec(config, int1e, i=0, a=0)
    h = 0 #={{{=#
    if i == 0
        #sum over all electrons (diag term)
        #println("diag term in one elec")
        for m in config
            h += int1e[m, m]
        end
    else
        #println("single excit in one elec")
        #only get single excitation contributation
        h += int1e[i,a]
    end#=}}}=#
    return h
end

function two_elec(config, int2e, i=0, a=0, j=0, b=0)
    #Anti symmetry included! compute the two electron contribution{{{
    g = 0
    
    #Diagonal Terms
    if i == 0 && a == 0 && j == 0 && b == 0 
        #sum over all electrons (diag term)
        #println("diagonal in two e")
        for n = 1:size(config)[1]
            for m = n+1:size(config)[1]
                g += int2e[config[n],config[n],config[m],config[m]]
                g -= int2e[config[n],config[m],config[n],config[m]]
            end
        end
    end

    #Single Excitation Terms
    if i!= 0 && a!=0 && j==0
        #single excitation 
        #println("single")
        for m in config
            if m == i
                continue
            else
                g += int2e[m,m,i,a]
                g -= int2e[m,i,m,a]
                #from olsen paper i=k and a=l
            end
        end
    end

    if i!= 0 && a!=0 && j!=0 && b!=0
        #double excitation
        #println("double in two elec")
        g += int2e[i, a, j, b]
        g -= int2e[i, j, a, b]
    end#=}}}=#
    return g
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


function old_get_H_mixed(configs, nelec, norbs, y_matrix, int2e)
    #configs = [alpha_configs, beta_configs]{{{
    #nelec = [n_alpha, n_beta]
    #y_matrix = [y_alpha, y_beta]
    ndets_a = factorial(norbs) / (factorial(nelec[1])*factorial(norbs-nelec[1])) 
    ndets_b = factorial(norbs) / (factorial(nelec[2])*factorial(norbs-nelec[2])) 
    size_mixed = Int(ndets_a*ndets_b)
    Hmixed = zeros(size_mixed, size_mixed)
    alpha_dets = size(configs[1])[1]

    for I in configs[1]  #alpha configs bra alpha
        orbs = [1:norbs;]
        vir_I = filter!(x->!(x in I), orbs)
        I_idx = get_index(I, y_matrix[1], norbs)
        for i in I
            for a in vir_I      #ket alpha
                for J in configs[2]     #beta configs bra beta
                    orbs2 = [1:norbs;]
                    vir_J = filter!(x->!(x in J), orbs2)
                    J_idx = get_index(J, y_matrix[2], norbs)
                    for j in J
                        for b in vir_J  #ket beta
                            #i think ill have some double counting in these
                            I_prim, sort_I, signi = excit_config(deepcopy(I), [i, a])
                            J_prim, sort_J, signj = excit_config(deepcopy(J), [j, b])
                            I_prim_idx = get_index(I_prim, y_matrix[1], norbs)
                            J_prim_idx = get_index(J_prim, y_matrix[2], norbs)
                            row_idx = Int(J_idx*alpha_dets + I_idx)
                            column_idx = Int(J_prim_idx*alpha_dets + I_prim_idx)
                            #println("row : ", row_idx, "column: ", column_idx)
                            #println("Index I: ", I_prim_idx, " Index J: ", J_prim_idx)
                            #println("two electron comp iajb: ", int2e[i,a,j,b])
                            #println("other two electron comp ijab: ", int2e[i,j,a,b])
                            Hmixed[row_idx, column_idx] += signi*signj*int2e[i, j, a, b]
                            #Hmixed[column_idx, row_idx] += Hmixed[row_idx, column_idx]
                        end
                    end
                end
            end
        end
    end#=}}}=#
    return Hmixed
end
