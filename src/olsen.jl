using fci
using LinearAlgebra
using Printf
using PyCall
using NPZ
using StaticArrays
using JLD2

struct H
    h1::Array
    h2::Array
end

function load_ints()
    @load "/Users/nicole/code/fci/test/data/_testdata_h8_integrals.jld2"
    return H(int1e, int2e)
end


function build_full_Hmatrix(ints::H, p::FCIProblem)
    Hmat = zeros(p.dim, p.dim)
    #if closed shell only compute single spin
    if p.na == p.nb 
        a_configs = compute_configs(p)[1]
        
        #fill single excitation lookup tables
        a_lookup = fill_lookup(p, a_configs)
        
        #compute diag terms of sigma 1 and sigma 2
        Ha_diag = precompute_spin_diag_terms(a_configs, p.na, p.dima, ints)
        
        #compute off diag terms of sigma1 and sigma2
        Ha = compute_ss_terms_full(a_configs, a_lookup, p.dima, p.no, p.na, ints) + Ha_diag
        
        Hmat .+= kron(SMatrix{p.dimb, p.dimb, UInt8}(Matrix{UInt8}(I,p.dimb,p.dimb)),  Ha)
        Hmat .+= kron(Ha, SMatrix{p.dima, p.dima, UInt8}(Matrix{UInt8}(I,p.dima,p.dima)))
        
        #compute both diag and off diag terms for sigma3 (mixed spin sigma)
        Hmat .+= compute_ab_terms_full(ints, p, a_configs, a_configs, a_lookup, a_lookup)
        eig = eigen(Hmat)
        println(eig.values[1:7])
        error("stop")
    
    #if open shell must compute alpha and beta separately
    else 
        a_configs = compute_configs(p)[1]
        b_configs = compute_configs(p)[2]
    
        #fill single excitation lookup tables
        a_lookup = fill_lookup(p, a_configs)
        b_lookup = fill_lookup(p, b_configs)

        #compute diag terms of sigma 1 and sigma 2
        Ha_diag = precompute_spin_diag_terms(a_configs, p.na, p.dima, ints)
        Hb_diag = precompute_spin_diag_terms(b_configs, p.nb, p.dimb, ints)
        
        #compute off diag terms of sigma1 and sigma2
        Ha = compute_ss_terms_full(a_configs, a_lookup, p.dima, p.no, p.na, ints) + Ha_diag
        Hb = compute_ss_terms_full(b_configs, b_lookup, p.dimb, p.no, p.nb, ints) + Hb_diag
        
        Hmat .+= kron(SMatrix{p.dimb, p.dimb, UInt8}(Matrix{UInt8}(I,p.dimb,p.dimb)),  Ha)
        Hmat .+= kron(Hb, SMatrix{p.dima, p.dima, UInt8}(Matrix{UInt8}(I,p.dima,p.dima)))
        
        #compute both diag and off diag terms for sigma3 (mixed spin sigma)
        Hmat .+= compute_ab_terms_full(ints, p, a_configs, b_configs, a_lookup, b_lookup)
    end

    return Hmat
end

function sigma_one(configs, norbs, nelecs, dim, ndim_a, dim_b, lookup, ci_vector, ints::H)
    #dim is full dim of CI vector
    sigma_out = zeros(ndim_a, dim_b)

    h1eff = deepcopy(ints.h1)
    @tensor begin
        h1eff[p,q] -= .5 * ints.h2[p,j,j,q]
    end
    
    for I in configs
        I_idx = I[2]
        I_config = I[1]
        orbs = [1:norbs;]
        vir_I = filter!(x->(x in I_config), orbs)
        I_idx_lin = I_idx + (I_idx-1)*ndim_a
        F = zeros(ndim_a)
        
        ##diag part
        for i in 1:nelecs
            F[I_idx] += ints.h1[I_config[i], I_config[i]]
            for j in i+1:nelecs
                F[I_idx] += ints.h2[I_config[i], I_config[i], I_config[j], I_config[j]]
                F[I_idx] -= ints.h2[I_config[i], I_config[j], I_config[i], I_config[j]]
            end
        end
        

        ##diag part
        #for i in 1:nelecs
        #    sigma_out[I_idx_lin] += ints.h1[I_config[i], I_config[i]]*vector[I_idx_lin]
        #    for j in i+1:nelec
        #        sigma_out[I_idx_lin] += ints.h2[I_config[i], I_config[i], I_config[j], I_config[j]]*vector[I_idx_lin]
        #        sigma_out[I_idx_lin] -= ints.h2[I_config[i], I_config[j], I_config[i], I_config[j]]*vector[I_idx_lin]
        #    end
        #end

        #single excitation
        for k in I_config
            for l in vir
                K_idx = lookup[k,l,I_idx]
                sign_s = sign(K_idx)
                F[abs(K_idx)] += sign_s*h1eff[k,l]
                #double index
                for i in I_config
                    for j in vir
                        single, sorted_s, sign_s = excit_config(I_config, k,l)
                        double, sorted_d, sign_d = excit_config(sorted_s, i,j)
                        J_idx = configs[sorted_d]
                        F[J_idx] += 0.5 * sign_s * sign_d * ints.h2[i,j,k,l]
                    end
                end
            end
        end
        sigma_out[:, I_idx] = ci_vector*F

    return sigma_out
end

function precompute_spin_diag_terms(configs, nelecs, dim, ints::H)
    Hout = zeros(dim, dim)
    for I in configs
        config = I[1]
        idx = I[2]
        for i in 1:nelecs
            Hout[idx, idx] += ints.h1[config[i], config[i]]
            for j in i+1:nelecs
                Hout[idx,idx] += ints.h2[config[i], config[i], config[j], config[j]]
                Hout[idx,idx] -= ints.h2[config[i], config[j], config[i], config[j]]
            end
        end
    end
    return Hout
end

function compute_ss_terms_full(configs, lookup, dim, norbs, nelecs, ints::H)
    Ha = zeros(dim, dim)
    for I in configs
        config = I[1]
        I_idx = I[2]
        F = zeros(dim)
        orbs = [1:norbs;]
        vir = filter!(x->!(x in config), orbs)
        
        #single excitation
        for k in config
            for l in vir
                single_idx = lookup[k,l,I_idx]
                sign_s = sign(single_idx)
                F[abs(single_idx)] += ints.h1[k,l]*sign_s
                for m in config
                    if m!=k
                        F[abs(single_idx)] += sign_s*(ints.h2[k,l,m,m]-ints.h2[k,m,l,m])
                    end
                end
            end
        end

        #double excitation
        for k in config
            for i in config
                if i>k
                    for l in vir
                        for j in vir
                            if j>l
                                single, sorted_s, sign_s = excit_config(config, k,l)
                                double, sorted_d, sign_d = excit_config(sorted_s, i,j)
                                idx = configs[sorted_d]
                                if sign_d == sign_s
                                    F[idx] += (ints.h2[i,j,k,l] - ints.h2[i,l,j,k]) 

                                else
                                    F[idx] -= (ints.h2[i,j,k,l] - ints.h2[i,l,j,k]) 
                                end
                            end
                        end
                    end
                end
            end
        end
        Ha[:,I_idx] .= F
    end
    return Ha
end

function compute_ab_terms_full(ints::H, prob::FCIProblem, a_configs, b_configs, a_lookup, b_lookup)
    Hmat = zeros(prob.dim, prob.dim)
    
    for Ka in a_configs
        Ka_idx = Ka[2]
        Ka_config = Ka[1]
        orbsa = [1:prob.no;]
        vira = filter!(x->!(x in Ka_config), orbsa)
        for Kb in b_configs
            Kb_idx = Kb[2]
            Kb_config = Kb[1]
            orbsb = [1:prob.no;]
            virb = filter!(x->!(x in Kb_config), orbsb)
            K = Ka_idx + (Kb_idx-1)*prob.dima #works for closed shell
            
            #diagonal part
            for l in Kb_config
                for n in Ka_config
                    Hmat[K, K] += ints.h2[n,n,l,l]
                end
            end
            
            #excit alpha only
            for p in Ka_config
                for q in vira
                    a_single, sort_a, sign_a = excit_config(deepcopy(Ka_config), p,q)
                    idxa = abs(a_lookup[p,q,Ka_idx])
                    Kprime = idxa + (Kb_idx-1)*prob.dima
                    #alpha beta <ii|jj>
                    for m in Kb_config
                        Hmat[K,Kprime]+=sign_a*ints.h2[p,q,m,m]
                    end
                end
            end

            #excit beta only
            for r in Kb_config
                for s in virb
                    b_single, sort_b, sign_b = excit_config(deepcopy(Kb_config), r,s)
                    idxb = abs(b_lookup[r,s,Kb_idx])
                    Lprime = Ka_idx + (idxb-1)*prob.dima
                    
                    #alpha beta <ii|jj>
                    for n in Ka_config
                        Hmat[K,Lprime]+=sign_b*ints.h2[r,s,n,n]
                    end
                end
            end

            #excit alpha and beta
            for p in Ka_config
                for q in vira
                    a_single, sort_a, sign_a = excit_config(deepcopy(Ka_config), p,q)
                    idxa = abs(a_lookup[p,q,Ka_idx])
                    for r in Kb_config
                        for s in virb
                            b_single, sort_b, sign_b = excit_config(deepcopy(Kb_config), r,s)
                            idxb = abs(b_lookup[r,s,Kb_idx])
                            L = idxa + (idxb-1)*prob.dima
                            Hmat[K,L] += sign_a*sign_b*(ints.h2[p,q,r,s])
                        end
                    end
                end
            end
        end
    end#=}}}=#
    return Hmat
end
    
        
function compute_configs(p::FCIProblem)
    ### Get_all_configs function will be replaced by GRMS depth first search function that will also compute the index and save as a vector of vectors{{{
    xalpha, yalpha = make_xy(p.no, p.na)

    if p.na == p.nb
        a_configs = get_all_configs(xalpha, yalpha, p.na)
        return (a_configs, a_configs)
    else
        xbeta, ybeta = make_xy(p.no, p.nb)
        a_configs = get_all_configs(xalpha, yalpha, p.na)
        b_configs = get_all_configs(xbeta, ybeta, p.nb)
        return (a_configs, b_configs)
    end#=}}}=#
end

function get_all_configs(x, y, nelecs)
    vert_graph, max_vert = fci.make_vert_graph(x)#={{{=#
    graph_dict = fci.make_graph_dict(y, vert_graph)
    config_dict = fci.dfs(nelecs, graph_dict, 1, max_vert)
    return config_dict#=}}}=#
end


function make_xy(norbs, nelec)
    #makes y matrices for grms indexing{{{
    n_unocc_a = (norbs-nelec)+1

    #make x matricies
    x = zeros(Int, n_unocc_a, nelec+1)
    #fill first row and columns
    x[:,1] .= 1
    x[1,:] .= 1
    
    for i in 2:nelec+1
        for j in 2:n_unocc_a
            x[j, i] = x[j-1, i] + x[j, i-1]
        end
    end

    #make y matrices
    y = vcat(transpose(zeros(Int, nelec+1)), x[1:n_unocc_a-1, :])#=}}}=#
    return x, y
end

function fill_lookup(p::FCIProblem, configs)
    lookup_table = zeros(Int64,p.no, p.no, p.dim)#={{{=#
    orbs = [1:p.no;]
    for i in configs
        vir = filter!(x->!(x in i[1]), [1:p.no;])
        for p in i[1]
            for q in vir
                new_config, sorted_config, sign_s = excit_config(i[1], p, q)
                idx = configs[sorted_config]
                lookup_table[p,q,i[2]] = sign_s*idx
            end
        end
    end#=}}}=#
    return lookup_table
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
    len = length(arr) #={{{=#
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




    
        
