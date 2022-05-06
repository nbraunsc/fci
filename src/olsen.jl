using fci
using LinearAlgebra
using Printf
using PyCall
using NPZ
using StaticArrays
using TensorOperations
using JLD2
using BenchmarkTools

struct H
    h1::Array
    h2::Array
end

function load_ints()
    #@load "/Users/nicole/code/fci/test/data/_testdata_h8_integrals.jld2"
    @load "/Users/nicole/code/fci/test/data/_testdata_h4_triplet_integrals.jld2"
    return H(one, two)
    #return H(int1e, int2e)
end

function run_fci(ints::H, p::FCIProblem, ci_vector=nothing, max_iter=12, nroots=1, tol=1e-6, precompute_ss=false)
    np = pyimport("numpy")
    e_vals = npzread("/Users/nicole/code/fci/src/data/eigenvals_elec.npy")
    a_configs = compute_configs(p)[1]
    b_configs = compute_configs(p)[2]
    
    #fill single excitation lookup tables
    a_lookup = fill_lookup(p, a_configs, p.dima)
    b_lookup = fill_lookup(p, b_configs, p.dimb)

    if precompute_ss
        Ha_diag = precompute_spin_diag_terms(a_configs, p.na, p.dima, ints)
        Hb_diag = precompute_spin_diag_terms(b_configs, p.nb, p.dimb, ints)
        
        #compute off diag terms of sigma1 and sigma2
        a_lookup_ov = fill_lookup_ov(p, a_configs, p.dima)
        b_lookup_ov = fill_lookup_ov(p, b_configs, p.dimb)
        Ha = compute_ss_terms_full(a_configs, a_lookup_ov, p.dima, p.no, p.na, ints) + Ha_diag
        Hb = compute_ss_terms_full(b_configs, b_lookup_ov, p.dimb, p.no, p.nb, ints) + Hb_diag
        
        e = fci.Lanczos(p, ints, a_configs, b_configs, a_lookup, b_lookup, Ha, Hb, ci_vector, max_iter, tol)
        println("Eigenvalue: ", e)
        println("Eigenvalues from pyscf: ", e_vals)
        
        #Hmat = compute_ab_terms_full(ints, p, a_configs, b_configs, a_lookup, b_lookup){{{
        #Ia = SMatrix{p.dima, p.dima, UInt8}(Matrix{UInt8}(I, p.dima, p.dima))
        #Ib = SMatrix{p.dimb, p.dimb, UInt8}(Matrix{UInt8}(I, p.dimb, p.dimb))
        #sigma1 = kron(Hb, Ia)*vec(ci_vector)
        #sigma2 = kron(Ib, Ha)*vec(ci_vector)
        #sigma3 = Hmat*vec(ci_vector)
        
        #@time sigma_one = compute_sigma_one_all(b_configs, b_lookup, ci_vector, ints, p)
        #@time sigma_two = compute_sigma_two_all(a_configs, a_lookup, ci_vector, ints, p)
        #@time sigma_three = compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci_vector, ints, p)
        #diff1 = sigma1-vec(sigma_one)
        #diff2 = sigma2-vec(sigma_two)
        #diff3 = sigma3-vec(sigma_three)
        #Base.display(diff2)}}}

    else
        e = fci.Lanczos(p, ints, a_configs, b_configs, a_lookup, b_lookup, ci_vector, max_iter, tol)
        println("Eigenvalue: ", e)
        println("Eigenvalues from pyscf: ", e_vals)

        #@time sigma_one = compute_sigma_one_all(b_configs, b_lookup, ci_vector, ints, p){{{
        #@time sigma_two = compute_sigma_two_all(a_configs, a_lookup, ci_vector, ints, p)
        #@time sigma_three = compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci_vector, ints, p)
        #@btime sigma_two = compute_sigma_two_all($a_configs, $a_lookup, $ci_vector, $ints, $p)
        
        #a_lookup_ov = fill_lookup_ov(p, a_configs, p.dima)
        #b_lookup_ov = fill_lookup_ov(p, b_configs, p.dimb)
        #@time sigma_one_ov = compute_sigma_one_ov(b_configs, b_lookup_ov, ci_vector, ints, p)
        #@btime sigma_two_ov = compute_sigma_two_ov($a_configs, $a_lookup_ov, $ci_vector, $ints, $p)
        #a_lookup_ov = fill_lookup_ov(p, a_configs, p.dima)
        #b_lookup_ov = fill_lookup_ov(p, b_configs, p.dimb)
        #@time sigma_three_ov = get_sigma3_unvec(a_configs, b_configs, ci_vector, a_lookup_ov, b_lookup_ov, ints, p)
        #mat_sigma3 = reshape(sigma_three_ov, p.dima, p.dimb) }}}
    end
    return e
end

function matvec(a_configs, b_configs, a_lookup, b_lookup, ints::H, p::FCIProblem, Ha, Hb, ci_vector=nothing)
    Ia = SMatrix{p.dima, p.dima, UInt8}(Matrix{UInt8}(I, p.dima, p.dima))
    Ib = SMatrix{p.dimb, p.dimb, UInt8}(Matrix{UInt8}(I, p.dimb, p.dimb))
    sigma1 = kron(Hb, Ia)*ci_vector
    sigma2 = kron(Ib, Ha)*ci_vector
    sigma3 = vec(compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci_vector, ints, p))
    sigma = sigma1 + sigma2 + sigma3
    return sigma
end

function matvec(a_configs, b_configs, a_lookup, b_lookup, ints::H, p::FCIProblem, ci_vector=nothing)
    sigma1 = vec(compute_sigma_one_all(b_configs, b_lookup, ci_vector, ints, p))
    sigma2 = vec(compute_sigma_two_all(a_configs, a_lookup, ci_vector, ints, p))
    sigma3 = vec(compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci_vector, ints, p))
    sigma = sigma1 + sigma2 + sigma3
    return sigma
end


function build_full_Hmatrix(ints::H, p::FCIProblem)
    Hmat = zeros(p.dim, p.dim)#={{{=#
    np = pyimport("numpy")
    e_vals = npzread("/Users/nicole/code/fci/src/data/eigenvals_elec.npy")
    #if closed shell only compute single spin
    if p.na == p.nb 
        a_configs = compute_configs(p)[1]
        
        #fill single excitation lookup tables
        a_lookup = fill_lookup_ov(p, a_configs, p.dima)
        
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
        println("Eigenvalues from pyscf: ", e_vals)
        error("stop")
    
    #if open shell must compute alpha and beta separately
    else 
        a_configs = compute_configs(p)[1]
        b_configs = compute_configs(p)[2]
    
        #fill single excitation lookup tables
        a_lookup = fill_lookup_ov(p, a_configs, p.dima)
        b_lookup = fill_lookup_ov(p, b_configs, p.dimb)

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
    end#=}}}=#
    return Hmat
end

function compute_sigma_one_ov(b_configs, b_lookup, ci_vector, ints::H, prob::FCIProblem)
    #dim is full dim of CI vector{{{
    sigma_one = zeros(prob.dima, prob.dimb)
    
    ## bb σ1(Iα, Iβ)
    for I_b in b_configs
        I_idx = I_b[2]
        I_config = I_b[1]
        orbs = [1:prob.no;]
        vir = filter!(x->!(x in I_config), orbs)
        #I_idx_lin = I_idx + (I_idx-1)*prob.dima
        F = zeros(prob.dimb)
        
        ##diag part
        for i in 1:prob.nb
            F[I_idx] += ints.h1[I_config[i], I_config[i]]
            for j in i+1:prob.nb
                F[I_idx] += ints.h2[I_config[i], I_config[i], I_config[j], I_config[j]]
                F[I_idx] -= ints.h2[I_config[i], I_config[j], I_config[i], I_config[j]]
            end
        end
        
        ##single excitation
        for k in I_config
            for l in vir
                K_idx = b_lookup[k,l,I_idx]
                sign_s = sign(K_idx)
                F[abs(K_idx)] += sign_s*ints.h1[k,l]
                for m in I_config
                    if m!=k
                        F[abs(K_idx)] += sign_s*(ints.h2[k,l,m,m]-ints.h2[k,m,l,m])
                    end
                end
            end
        end
                
        ##double excitation
        for k in I_config
            for i in I_config
                if i>k
                    for l in vir
                        for j in vir
                            if j>l
                                single, sorted_s, sign_s = excit_config(I_config, k,l)
                                double, sorted_d, sign_d = excit_config(sorted_s, i,j)
                                J_idx = b_configs[sorted_d]
                                if sign_d == sign_s
                                    F[abs(J_idx)] += (ints.h2[i,j,k,l] - ints.h2[i,l,j,k]) #one that works for closed

                                else
                                    F[abs(J_idx)] -= (ints.h2[i,j,k,l] - ints.h2[i,l,j,k]) #one that works for closed
                                end
                            end
                        end
                    end
                end
            end
        end
        mat = Diagonal(F)
        sigma_one += mat*transpose(ci_vector)
    end
    return sigma_one #=}}}=#
end

function compute_sigma_two_ov(a_configs, a_lookup, ci_vector, ints::H, prob::FCIProblem)
    sigma_two = zeros(prob.dima, prob.dimb)#={{{=#

    ## aa σ2(Iα, Iβ)
    for I in a_configs
        I_idx = I[2]
        I_config = I[1]
        orbs = [1:prob.no;]
        vir = filter!(x->!(x in I_config), orbs)
        #I_idx_lin = I_idx + (I_idx-1)*prob.dima
        F = zeros(prob.dima)
        
        ##diag part
        for i in 1:prob.na
            F[I_idx] += ints.h1[I_config[i], I_config[i]]
            for j in i+1:prob.na
                F[I_idx] += ints.h2[I_config[i], I_config[i], I_config[j], I_config[j]]
                F[I_idx] -= ints.h2[I_config[i], I_config[j], I_config[i], I_config[j]]
            end
        end
        
        #single excitation
        for k in I_config
            for l in vir
                K_idx = a_lookup[k,l,I_idx]
                sign_s = sign(K_idx)
                F[abs(K_idx)] += sign_s*ints.h1[k,l]
                for m in I_config
                    if m!=k
                        F[abs(K_idx)] += sign_s*(ints.h2[k,l,m,m]-ints.h2[k,m,l,m])
                    end
                end
            end
        end

        #double excitation
        for k in I_config
            for i in I_config
                if i > k
                    for l in vir
                        for j in vir
                            if j > l
                                single, sorted_s, sign_s = excit_config(I_config, k,l)
                                double, sorted_d, sign_d = excit_config(sorted_s, i,j)
                                J_idx = a_configs[sorted_d]
                                if sign_d == sign_s
                                    F[abs(J_idx)] += (ints.h2[i,j,k,l] - ints.h2[i,l,j,k]) #one that works for closed

                                else
                                    F[abs(J_idx)] -= (ints.h2[i,j,k,l] - ints.h2[i,l,j,k]) #one that works for closed
                                end
                            end
                        end
                    end
                end
            end
        end
        mat = Diagonal(F)
        sigma_two += mat*ci_vector
    end
    return sigma_two#=}}}=#
end

function compute_sigma_one_all(b_configs, b_lookup, ci_vector, ints::H, prob::FCIProblem)
    ## bb σ1(Iα, Iβ){{{
    sigma_one = zeros(prob.dima, prob.dimb)
    ci_vector = reshape(ci_vector, prob.dima, prob.dimb)
    
    h1eff = deepcopy(ints.h1)
    @tensor begin
        h1eff[p,q] -= .5 * ints.h2[p,j,j,q]
    end
    
    for I_b in b_configs
        I_idx = I_b[2]
        I_config = I_b[1]
        F = zeros(prob.dimb)
        
        ##single excitation
        for k in 1:prob.no
            for l in 1:prob.no
                K_idx = b_lookup[k,l,I_idx]
                if K_idx == 0
                    continue
                end
                sign_kl = sign(K_idx)
                K = abs(K_idx)
                F[K] += sign_kl*h1eff[k,l]
               
                #double excitation
                for i in 1:prob.no
                    for j in 1:prob.no
                        J_idx = b_lookup[i,j,K]
                        if J_idx == 0 
                            continue
                        end
                        sign_ij = sign(J_idx)
                        J = abs(J_idx)
                        if sign_kl == sign_ij
                            F[J] += 0.5*ints.h2[i,j,k,l]
                        else
                            F[J] -= 0.5*ints.h2[i,j,k,l]
                        end
                    end
                end
            end
        end
        #mat = Diagonal(F)
        #sigma_one += transpose(mat*transpose(ci_vector))
         
        for Ja in 1:prob.dimb
            for Kb in 1:prob.dima
                sigma_one[Kb, I_idx] += F[Ja]*ci_vector[Kb,Ja]
            end
        end

        

    end#=}}}=#
    return sigma_one
end

function compute_sigma_two_all(a_configs, a_lookup, ci_vector, ints::H, prob::FCIProblem)
    ## aa σ2(Iα, Iβ){{{
    sigma_two = zeros(prob.dima, prob.dimb)
    ci_vector = reshape(ci_vector, prob.dima, prob.dimb)

    h1eff = deepcopy(ints.h1)
    @tensor begin 
        h1eff[p,q] -= .5 * ints.h2[p,j,j,q]
    end
    
    for I_a in a_configs
        I_idx = I_a[2]
        I_config = I_a[1]
        F = zeros(prob.dima)
        
        #single excitation
        for k in 1:prob.no
            for l in 1:prob.no
                K_idx = a_lookup[k,l,I_idx]
                if K_idx == 0
                    continue
                end
                sign_kl = sign(K_idx)
                K = abs(K_idx)
                F[K] += sign_kl*h1eff[k,l]
                
                #double excitation
                for i in 1:prob.no
                    for j in 1:prob.no
                        J_idx = a_lookup[i,j,K]
                        if J_idx == 0 
                            continue
                        end
                        sign_ij = sign(J_idx)
                        J = abs(J_idx)
                        if sign_kl == sign_ij
                            F[J] += 0.5*ints.h2[i,j,k,l]
                        else
                            F[J] -= 0.5*ints.h2[i,j,k,l]
                        end
                    end
                end
            end
        end
        #mat = Diagonal(F)
        #sigma_two += mat*ci_vector
        
        for Ja in 1:prob.dima
            for Kb in 1:prob.dimb
                sigma_two[I_idx, Kb] += F[Ja]*ci_vector[Ja,Kb]
            end
        end

    end#=}}}=#
    return sigma_two
end

function compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci_vector, ints::H, prob::FCIProblem)
    sigma_three = zeros(prob.dima, prob.dimb)#={{{=#
    ci_vector = reshape(ci_vector, prob.dima, prob.dimb)
    for k in 1:prob.no
        for l in 1:prob.no
            L = Vector{Int32}()
            R = Vector{Int32}()
            sign_I = Vector{Int32}()
            # compute all k->l excitations of alpha ket
            # config R -> config L with a k->l excitation
            for I in a_configs
                Iidx = a_lookup[k,l,I[2]]
                if Iidx != 0
                    push!(R,I[2])
                    push!(L,abs(Iidx))
                    push!(sign_I, sign(Iidx))
                end
            end
            
            #Gather
            # .* is vectorized multiplication
            Ckl = ci_vector[L, :]
            Ckl_prime = sign_I .* Ckl

            #look over beta configs
            for Ib in b_configs
                F = zeros(prob.dimb)

                #loop over i->j excitations of beta ket
                for i in 1:prob.no
                    for j in 1:prob.no
                        Jb = b_lookup[i,j,Ib[2]]
                        if Jb == 0
                            continue
                        end
                        sign_ij = sign(Jb)
                        J = abs(Jb)
                        F[J] += sign_ij*ints.h2[i,j,k,l]
                    end
                end
                
                #VI = Ckl*F.*sign_I
                VI = Ckl_prime*F
                
                #Scatter
                for Li in 1:length(VI)
                    sigma_three[R[Li], Ib[2]] += VI[Li]
                end
            end
        end
    end
    return sigma_three#=}}}=#
end

function get_sigma3_unvec(a_configs, b_configs, vector, a_lookup, b_lookup, ints::H, prob::FCIProblem)
    sigma3 = zeros(prob.dim) #{{{

    for I in a_configs  #alpha configs
        orbs = [1:prob.no;] ## change this to SVector
        vir_I = filter!(x->!(x in I[1]), orbs)
        I_idx = I[2]
        I_config = I[1]
        for J in b_configs     #beta configs
            orbs2 = [1:prob.no;]
            vir_J = filter!(x->!(x in J[1]), orbs2)
            J_idx = J[2]
            J_config = J[1]
            K = I_idx + (J_idx-1)*prob.dima

            for l in J_config
                for n in I_config
                    #Hmat[K,K]
                    sigma3[K] += ints.h2[n,n,l,l]*vector[K]
                end
            end

            #excit alpha only
            for p in I_config
                for q in vir_I
                    idxa = a_lookup[p,q,I_idx]
                    sign_a = sign(idxa)
                    Kprim = abs(idxa) + (J_idx-1)*prob.dima
                    for m in J_config
                        #Hmat[K,Kprim]
                        sigma3[K]+=sign_a*ints.h2[p,q,m,m]*vector[Kprim]
                    end
                end
            end

            #excit beta only
            for r in J_config
                for s in vir_J
                    idxb = b_lookup[r,s,J_idx]
                    sign_b = sign(idxb)
                    Lprim = I_idx + (abs(idxb)-1)*prob.dima
                    for n in I_config
                        #Hmat[K,Lprim]
                        sigma3[K]+=sign_b*ints.h2[r,s,n,n]*vector[Lprim]
                    end
                end
            end

            #excit both alpha and beta
            for p in I_config
                for q in vir_I
                    idxa = a_lookup[p,q,I_idx]
                    sign_a = sign(idxa)
                    for r in J_config
                        for s in vir_J
                            idxb = b_lookup[r,s,J_idx]
                            sign_b = sign(idxb)
                            L = abs(idxa) + (abs(idxb)-1)*prob.dima
                            #Hmat[K,L]
                            sigma3[K] += sign_a*sign_b*ints.h2[p,q,r,s]*vector[L]
                        end
                    end
                end
            end
        end
    end
    return sigma3#=}}}=#
end


function precompute_spin_diag_terms(configs, nelecs, dim, ints::H)
    Hout = zeros(dim, dim)#={{{=#
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
    end#=}}}=#
    return Hout
end

function compute_ss_terms_full(configs, lookup, dim, norbs, nelecs, ints::H)
    Ha = zeros(dim, dim)#={{{=#
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
    end#=}}}=#
    return Ha
end

function compute_ab_terms_full(ints::H, prob::FCIProblem, a_configs, b_configs, a_lookup, b_lookup)
    Hmat = zeros(prob.dim, prob.dim)#={{{=#
    
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
                    a_single, sort_a, sign_a = excit_config(Ka_config, p,q)
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
                    b_single, sort_b, sign_b = excit_config(Kb_config, r,s)
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
                    a_single, sort_a, sign_a = excit_config(Ka_config, p,q)
                    idxa = abs(a_lookup[p,q,Ka_idx])
                    for r in Kb_config
                        for s in virb
                            b_single, sort_b, sign_b = excit_config(Kb_config, r,s)
                            idxb = abs(b_lookup[r,s,Kb_idx])
                            L = idxa + (idxb-1)*prob.dima
                            Hmat[K,L] += sign_a*sign_b*(ints.h2[p,q,r,s])
                        end
                    end
                end
            end
        end
    end#=}}}=#

    return Hmat#=}}}=#
end
    
function compute_configs(p::FCIProblem)
    ### {{{
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

function fill_lookup_ov(p::FCIProblem, configs, dim_s)
    lookup_table = zeros(Int64,p.no, p.no, dim_s)#={{{=#
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

function fill_lookup(prob::FCIProblem, configs, dim_s)
    lookup_table = zeros(Int64,prob.no, prob.no, dim_s)#={{{=#
    orbs = [1:prob.no;]
    for i in configs
        vir = filter!(x->!(x in i[1]), [1:prob.no;])
        for p in i[1]
            for q in 1:prob.no
                if p == q
                    lookup_table[p,q,i[2]] = i[2]
                end

                if q in vir
                    new_config, sorted_config, sign_s = excit_config(i[1], p, q)
                    idx = configs[sorted_config]
                    lookup_table[p,q,i[2]] = sign_s*idx
                end
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




    
        
