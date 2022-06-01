using LinearAlgebra
using Printf
using StaticArrays
using JLD2
using BenchmarkTools
using InteractiveUtils

using fci

function run_ras(ints::H, prob::RASProblem, precompute_ss=false, max_iter=12, ci_vector=nothing,  nroots=1, tol=1e-6)
    a_configs = ras_compute_configs(prob)[1]
    b_configs = ras_compute_configs(prob)[2]
    
    #fill single excitation lookup tables
    a_lookup = ras_fill_lookup(prob, a_configs, prob.dima)
    b_lookup = ras_fill_lookup(prob, b_configs, prob.dimb)
    
    if precompute_ss
        Ha_diag = fci.precompute_spin_diag_terms(a_configs, prob.na, prob.dima, ints)
        Hb_diag = fci.precompute_spin_diag_terms(b_configs, prob.nb, prob.dimb, ints)
        
        #compute off diag terms of sigma1 and sigma2
        #a_lookup_ov = ras_fill_lookup_ov(prob, a_configs, prob.dima)
        #b_lookup_ov = ras_fill_lookup_ov(prob, b_configs, prob.dimb)
        Ha = ras_compute_ss_terms_full(a_configs, a_lookup, prob.dima, prob.no, prob.na, ints) + Ha_diag
        Hb = ras_compute_ss_terms_full(b_configs, b_lookup, prob.dimb, prob.no, prob.nb, ints) + Hb_diag
        
        e = fci.Lanczos(prob, ints, a_configs, b_configs, a_lookup, b_lookup, Ha, Hb, ci_vector, max_iter, tol)
        println("Energy(Hartree): ", e)
        #println("Eigenvalues from pyscf: ", e_vals)
        
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
        #a_lookup_ov = ras_fill_lookup_ov(prob, a_configs, prob.dima)
        #b_lookup_ov = ras_fill_lookup_ov(prob, b_configs, prob.dimb)
        #e = fci.Lanczos(prob, ints, a_configs, b_configs, a_lookup_ov, b_lookup_ov, ci_vector, max_iter, tol)
        e = fci.Lanczos(prob, ints, a_configs, b_configs, a_lookup, b_lookup, ci_vector, max_iter, tol)
        println("Energy(Hartree): ", e)
        #println("Eigenvalues from pyscf: ", e_vals)

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

function build_full_Hmatrix(ints::H, p::RASProblem)
    Hmat = zeros(p.dim, p.dim)#={{{=#
    #np = pyimport("numpy")
    #e_vals = npzread("/Users/nicole/code/fci/src/data/eigenvals_elec.npy")
    #if closed shell only compute single spin
    if p.na == p.nb 
        a_configs = ras_compute_configs(p)[1]
        b_configs = ras_compute_configs(p)[2]

        #fill single excitation lookup tables
        a_lookup = ras_fill_lookup(p, a_configs, p.dima)
        b_lookup = ras_fill_lookup(p, b_configs, p.dimb)
        
        #compute diag terms of sigma 1 and sigma 2
        Ha_diag = fci.precompute_spin_diag_terms(a_configs, p.na, p.dima, ints)
        Hb_diag = fci.precompute_spin_diag_terms(b_configs, p.nb, p.dimb, ints)
        
        #compute off diag terms of sigma1 and sigma2
        #a_lookup_ov = ras_fill_lookup_ov(p, a_configs, p.dima)
        #b_lookup_ov = ras_fill_lookup_ov(p, b_configs, p.dimb)
        Ha = ras_compute_ss_terms_full(a_configs, a_lookup, p.dima, p.no, p.na, ints) + Ha_diag
        Hb = ras_compute_ss_terms_full(b_configs, b_lookup, p.dimb, p.no, p.nb, ints) + Hb_diag
        
        Hmat .+= kron(SMatrix{p.dimb, p.dimb, UInt8}(Matrix{UInt8}(I,p.dimb,p.dimb)),  Ha)
        Hmat .+= kron(Hb, SMatrix{p.dima, p.dima, UInt8}(Matrix{UInt8}(I,p.dima,p.dima)))
        
        #compute both diag and off diag terms for sigma3 (mixed spin sigma)
        Hmat .+= ras_compute_ab_terms_full(ints, p, a_configs, b_configs, a_lookup, b_lookup)
        Hmat = .5*(Hmat+Hmat')
        eig = eigen(Hmat)
        println(eig.values[1:7])
        #println("Eigenvalues from pyscf: ", e_vals)
    
    #if open shell must compute alpha and beta separately
    else 
        a_configs = compute_configs(p)[1]
        b_configs = compute_configs(p)[2]
    
        #fill single excitation lookup tables
        a_lookup = fill_lookup(p, a_configs, p.dima)
        b_lookup = fill_lookup(p, b_configs, p.dimb)
        #a_lookup = fill_lookup_ov(p, a_configs, p.dima)
        #b_lookup = fill_lookup_ov(p, b_configs, p.dimb)

        #compute diag terms of sigma 1 and sigma 2
        Ha_diag = fci.precompute_spin_diag_terms(a_configs, p.na, p.dima, ints)
        Hb_diag = fci.precompute_spin_diag_terms(b_configs, p.nb, p.dimb, ints)
        
        #compute off diag terms of sigma1 and sigma2
        Ha = ras_compute_ss_terms_full(a_configs, a_lookup, p.dima, p.no, p.na, ints) + Ha_diag
        Hb = ras_compute_ss_terms_full(b_configs, b_lookup, p.dimb, p.no, p.nb, ints) + Hb_diag
        
        Hmat .+= kron(SMatrix{p.dimb, p.dimb, UInt8}(Matrix{UInt8}(I,p.dimb,p.dimb)),  Ha)
        Hmat .+= kron(Hb, SMatrix{p.dima, p.dima, UInt8}(Matrix{UInt8}(I,p.dima,p.dima)))
        
        #compute both diag and off diag terms for sigma3 (mixed spin sigma)
        Hmat .+= ras_compute_ab_terms_full(ints, p, a_configs, b_configs, a_lookup, b_lookup)
        Hmat = .5*(Hmat+Hmat')
        eig = eigen(Hmat)
        println(eig.values[1:7])
    end#=}}}=#
    return Hmat
end

function matvec(a_configs, b_configs, a_lookup, b_lookup, ints::H, p::RASProblem, Ha, Hb, ci_vector=nothing)
    Ia = SMatrix{p.dima, p.dima, UInt8}(Matrix{UInt8}(I, p.dima, p.dima))#={{{=#
    Ib = SMatrix{p.dimb, p.dimb, UInt8}(Matrix{UInt8}(I, p.dimb, p.dimb))
    sigma1 = kron(Hb, Ia)*ci_vector
    sigma2 = kron(Ib, Ha)*ci_vector
    sigma3 = vec(compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci_vector, ints, p))
    sigma = sigma1 + sigma2 + sigma3
    return sigma#=}}}=#
end

function matvec(a_configs, b_configs, a_lookup, b_lookup, ints::H, p::RASProblem, ci_vector=nothing)
    sigma1 = vec(compute_sigma_one(b_configs, b_lookup, ci_vector, ints, p))#={{{=#
    sigma2 = vec(compute_sigma_two(a_configs, a_lookup, ci_vector, ints, p))
    sigma3 = vec(compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci_vector, ints, p))
    sigma = sigma1 + sigma2 + sigma3
    return sigma#=}}}=#
end

function compute_sigma_one(b_configs, b_lookup, ci_vector, ints::H, prob::RASProblem)
    ## bb σ1(Iα, Iβ){{{
    sigma_one = zeros(prob.dima, prob.dimb)
    ci_vector = reshape(ci_vector, prob.dima, prob.dimb)
    
    F = zeros(prob.dimb)
    hkl = zeros(prob.no, prob.no)
    
    h1eff = zeros(prob.no, prob.no)
    h1eff .= ints.h1
    #h1eff = deepcopy(ints.h1)
    @tensor begin
        h1eff[p,q] -= .5 * ints.h2[p,j,j,q]
    end
    
    for I_b in b_configs
        I_idx = I_b[2]
        I_config = I_b[1]
        fill!(F,0.0)

        ##single excitation
        for k in 1:prob.no
            for l in 1:prob.no
                K_idx = b_lookup[k,l,I_idx]
                if K_idx == 0
                    continue
                end
                sign_kl = sign(K_idx)
                K = abs(K_idx)
                @inbounds F[K] += sign_kl*h1eff[k,l]

                hkl .= ints.h2[:,:,k,l]
                _update_F_ss!(hkl, F, b_lookup, K, sign_kl, prob)

                #double excitation
#                for i in 1:prob.no
#                    for j in 1:prob.no
#                        J_idx = b_lookup[i,j,K]
#                        if J_idx == 0 
#                            continue
#                        end
#                        sign_ij = sign(J_idx)
#                        J = abs(J_idx)
#                        if sign_kl == sign_ij
#                            F[J] += 0.5*ints.h2[i,j,k,l]
#                        else
#                            F[J] -= 0.5*ints.h2[i,j,k,l]
#                        end
#                    end
#                end
            end
        end
        #mat = Diagonal(F)
        #sigma_one += transpose(mat*transpose(ci_vector))
         
        for Ja in 1:prob.dimb
            for Kb in 1:prob.dima
                @inbounds sigma_one[Kb, I_idx] += F[Ja]*ci_vector[Kb,Ja]
            end
        end
    end#=}}}=#
    return sigma_one
end

function compute_sigma_two(a_configs, a_lookup, ci_vector, ints::H, prob::RASProblem)
    ## aa σ2(Iα, Iβ){{{
    sigma_two = zeros(prob.dima, prob.dimb)
    ci_vector = reshape(ci_vector, prob.dima, prob.dimb)
    hkl = zeros(prob.no, prob.no)
    
    h1eff = zeros(prob.no, prob.no)
    h1eff .= ints.h1
    #h1eff = deepcopy(ints.h1)
    @tensor begin 
        h1eff[p,q] -= .5 * ints.h2[p,j,j,q]
    end
    
    F = zeros(prob.dima)
    for I_a in a_configs
        I_idx = I_a[2]
        I_config = I_a[1]
        fill!(F, 0.0)
        #single excitation
        for k in 1:prob.no
            for l in 1:prob.no
                K_idx = a_lookup[k,l,I_idx]
                if K_idx == 0
                    continue
                end
                sign_kl = sign(K_idx)
                K = abs(K_idx)
                @inbounds F[K] += sign_kl*h1eff[k,l]
                
                hkl .= ints.h2[:,:,k,l]
                _update_F_ss!(hkl, F, a_lookup, K, sign_kl, prob)
                #@code_warntype _update_F_ss!(hkl, F, a_lookup, K, sign_kl, prob)
                #error("here")
                
#                #double excitation
#                for j in 1:prob.no
#                    for i in 1:prob.no
#                        J_idx = a_lookup[i,j,K]
#                        if J_idx == 0 
#                            continue
#                        end
#                        sign_ij = sign(J_idx)
#                        J = abs(J_idx)
#                        if sign_kl == sign_ij
#                            F[J] += 0.5*ints.h2[i,j,k,l]
#                        else
#                            F[J] -= 0.5*ints.h2[i,j,k,l]
#                        end
#                    end
#                end
            end
        end
        #mat = Diagonal(F)
        #sigma_two += mat*ci_vector
        
        for Ja in 1:prob.dima
            for Kb in 1:prob.dimb
                @inbounds sigma_two[I_idx, Kb] += F[Ja]*ci_vector[Ja,Kb]
            end
        end

    end#=}}}=#
    #@save "src/data/h8_sigma2.jld2" sigma_two
    return sigma_two
end

function compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci_vector, ints::H, prob::RASProblem)
    sigma_three = zeros(prob.dima, prob.dimb)#={{{=#
    ci_vector = reshape(ci_vector, prob.dima, prob.dimb)
                
    F = zeros(prob.dimb)
    hkl = zeros(prob.no, prob.no)
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
                
            #@views hkl = ints.h2[:,:,k,l]

            hkl .= ints.h2[:,:,k,l]
            
            #look over beta configs
            for Ib in b_configs

                fill!(F,0.0)
                #@code_warntype _update_F!(hkl, F, b_lookup, Ib, prob)
                #error("here")
                #@views hkl = ints.h2[:,:,k,l]
                _update_F!(hkl, F, b_lookup, Ib[2], prob)
                
                #loop over i->j excitations of beta ket
#                for j in 1:prob.no
#                    jkl_idx = j-1 + (k-1)*prob.no + (l-1)*prob.no*prob.no 
#                    for i in 1:prob.no
#
#                        ijkl_idx = (i-1) + jkl_idx*prob.no + 1
#
#                        Jb = b_lookup[i,j,Ib[2]]
#                        if Jb == 0
#                            continue
#                        end
#                        sign_ij = sign(Jb)
#                        J = abs(Jb)
#                        #@inbounds F[J] += sign_ij*hkl[i,j]
#                        #@inbounds F[J] += sign_ij*ints.h2[ijkl_idx]
#                        F[J] += sign_ij*ints.h2[i,j,k,l]
#                    end
#                end
                
                #VI = Ckl*F.*sign_I
                VI = Ckl_prime*F
                
                #Scatter
                for Li in 1:length(VI)
                    sigma_three[R[Li], Ib[2]] += VI[Li]
                end
            end
        end
    end
#    @save "src/data/h8_sigma3_old.jld2" sigma_three
    return sigma_three#=}}}=#
end

function ras_compute_ss_terms_full(configs, lookup, dim, norbs, nelecs, ints::H)
    Ha = zeros(dim, dim)#={{{=#
    F = zeros(dim)
    for I in configs
        fill!(F, 0.0)
        config = I[1]
        I_idx = I[2]
        #orbs = [1:norbs;]
        vir = filter!(x->!(x in config), [1:norbs;])
        
        #single excitation
        for k in config
            for l in vir
                single_idx = lookup[k,l,I_idx]
                if single_idx == 0
                    continue
                end

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
                                if haskey(configs, sorted_d)
                                    idx_new = configs[sorted_d]
                                else
                                    continue
                                end
                                
                                if sign_d == sign_s
                                    @inbounds F[idx_new] += (ints.h2[i,j,k,l] - ints.h2[i,l,j,k]) 

                                else
                                    @inbounds F[idx_new] -= (ints.h2[i,j,k,l] - ints.h2[i,l,j,k]) 
                                end
                            end
                        end
                    end
                end
            end
        end
        Ha[:,I_idx] .= F
    end#=}}}=#
    #@save "src/data/h8_ha.jld2" Ha
    return Ha
end

function ras_compute_ab_terms_full(ints::H, prob::RASProblem, a_configs, b_configs, a_lookup, b_lookup)
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
            K = Ka_idx + (Kb_idx-1)*prob.dima
            
            #diagonal part
            for l in Kb_config
                for n in Ka_config
                    Hmat[K, K] += ints.h2[n,n,l,l]
                end
            end
            
            #excit alpha only
            for p in Ka_config
                for q in vira
                    #a_single, sort_a, sign_a = excit_config(Ka_config, p,q)
                    idxa = a_lookup[p,q,Ka_idx]
                    if idxa == 0
                        continue
                    end

                    sign_a = sign(idxa)
                    Kprime = abs(idxa) + (Kb_idx-1)*prob.dima
                    #alpha beta <ii|jj>
                    for m in Kb_config
                        Hmat[K,Kprime]+=sign_a*ints.h2[p,q,m,m]
                    end
                end
            end

            #excit beta only
            for r in Kb_config
                for s in virb
                    #b_single, sort_b, sign_b = excit_config(Kb_config, r,s)
                    idxb = b_lookup[r,s,Kb_idx]
                    if idxb == 0
                        continue
                    end

                    sign_b = sign(idxb)
                    Lprime = Ka_idx + (abs(idxb)-1)*prob.dima
                    
                    #alpha beta <ii|jj>
                    for n in Ka_config
                        Hmat[K,Lprime]+=sign_b*ints.h2[r,s,n,n]
                    end
                end
            end

            #excit alpha and beta
            for p in Ka_config
                for q in vira
                    #a_single, sort_a, sign_a = excit_config(Ka_config, p,q)
                    idxa = a_lookup[p,q,Ka_idx]
                    if idxa == 0
                        continue
                    end
                    sign_a = sign(idxa)
                    for r in Kb_config
                        for s in virb
                            #b_single, sort_b, sign_b = excit_config(Kb_config, r,s)
                            idxb = b_lookup[r,s,Kb_idx]
                            if idxb == 0
                                continue
                            end
                            sign_b = sign(idxb)
                            L = abs(idxa) + (abs(idxb)-1)*prob.dima
                            Hmat[K,L] += sign_a*sign_b*(ints.h2[p,q,r,s])
                        end
                    end
                end
            end
        end
    end#=}}}=#

    return Hmat#=}}}=#
end
    
function ras_fill_lookup(prob::RASProblem, configs, dim_s)
    lookup_table = zeros(Int64,prob.no, prob.no, dim_s)#={{{=#
    orbs = [1:prob.no;]
    ras1, ras3 = make_rasorbs(prob.fock[1], prob.fock[3], prob.no)
    for i in configs
        vir = filter!(x->!(x in i[1]), [1:prob.no;])
        for p in i[1]
            for q in 1:prob.no
                if p == q
                    lookup_table[p,q,i[2]] = i[2]
                end

                if q in vir
                    new_config, sorted_config, sign_s = fci.excit_config(i[1], p, q)
                    
                    #CHECK POINT for excitation
                    ras1_test = length(findall(in(sorted_config),ras1))
                    if ras1_test < prob.ras1_min
                        continue
                    end

                    ras3_test = length(findall(in(sorted_config),ras3))
                    if ras3_test > prob.ras3_max
                        continue
                    end
                    
                    idx = configs[new_config]
                    lookup_table[p,q,i[2]] = sign_s*idx
                end
            end
        end
    end#=}}}=#
    return lookup_table
end

function ras_fill_lookup_ov(prob::RASProblem, configs, dim_s)
    lookup_table = zeros(Int64,prob.no, prob.no, dim_s)#={{{=#
    orbs = [1:prob.no;]
    ras1, ras3 = make_rasorbs(prob.fock[1], prob.fock[3], prob.no)
    for i in configs
        vir = filter!(x->!(x in i[1]), [1:prob.no;])
        for p in i[1]
            for q in vir
                new_config, sorted_config, sign_s = excit_config(i[1], p, q)
                
                #CHECK POINT for excitation
                ras1_test = length(findall(in(sorted_config),ras1))
                if ras1_test < prob.ras1_min
                    continue
                end

                ras3_test = length(findall(in(sorted_config),ras3))
                if ras3_test > prob.ras3_max
                    continue
                end

                idx = configs[sorted_config]
                lookup_table[p,q,i[2]] = sign_s*idx
            end
        end
    end#=}}}=#
    return lookup_table
end
    
function make_rasorbs(rasi_orbs, rasiii_orbs, norbs, frozen_core=false)
    if frozen_core==false
        i_orbs = [1:1:rasi_orbs;]
        start = norbs-rasiii_orbs+1
        iii_orbs = [start:1:norbs;]
        return i_orbs, iii_orbs
    end
end

function ras_compute_configs(prob::RASProblem)
    yalpha = fci.make_ras_y(prob.xalpha, prob.no, prob.na, prob.fock, prob.ras1_min, prob.ras3_max)
    ybeta = fci.make_ras_y(prob.xbeta, prob.no, prob.nb, prob.fock, prob.ras1_min, prob.ras3_max)

    if prob.na == prob.nb
        a_configs = ras_get_all_configs(prob.xalpha, yalpha, prob.na)
        return (a_configs, a_configs)
    else
        a_configs = ras_get_all_configs(prob.xalpha, yalpha, prob.na)
        b_configs = ras_get_all_configs(prob.xbeta, ybeta, prob.nb)
        return (a_configs, b_configs)
    end
end

function ras_get_all_configs(x, y, nelecs)
    vert_graph, max_vert = fci.make_vert_graph_ras(x)
    graph_dict = fci.make_ras_dict(y, vert_graph)
    config_dict = fci.dfs_ras(nelecs, graph_dict, 1, max_vert)
    return config_dict
end



