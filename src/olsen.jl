#module diagonalize
#export diagonlize
using PyCall
using NPZ
using Printf
using Combinatorics

#orbs = 3
#nalpha = 2
#nbeta = 1
function diagonalize(orbs, nalpha, nbeta)
    #get eigenvalues from lanczos
    int1e = npzread("/Users/nicole/code/fci/src/data/int1e.npy")
    int2e = npzread("/Users/nicole/code/fci/src/data/int2e.npy")
    yalpha, ybeta = make_xy(orbs, nalpha, nbeta)
    
    #get all configs
    configa = zeros(orbs)
    configb = zeros(orbs)
    configa[1:nalpha] .= 1
    configb[1:nbeta] .= 1
    alpha_configs = get_all_configs(configa)
    beta_configs = get_all_configs(configb)

    #get H components
    H_alpha = get_H_same(alpha_configs, nalpha, orbs, yalpha, int1e, int2e)
    H_beta = get_H_same(beta_configs, nbeta, orbs, ybeta, int1e, int2e)
    H_mixed = ([alpha_configs, beta_configs], [nalpha, nbeta], orbs, [yalpha, ybeta], int2e)
   
    b = zeros(size(H_mixed)[1])
    b[1] = 1
    #initalize empty matrices for T and V
    T = zeros(Float64, m+1, m)
    V = zeros(Float64, size(matrix)[1], m+1)
    #normalize start vector
    V[:,1] = b/norm(b)
    #next vector
    w = get_sigma(H_alpha, H_beta, H_mixed, V[:,1])
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
    return Tm, V
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
    arraya = ones(size(xalpha)[2])
    arrayb = ones(size(xbeta)[2])
    yalpha = vcat(transpose(arraya), xalpha[1:size(xalpha)[1]-1, :])
    ybeta = vcat(transpose(arrayb), xbeta[1:size(xbeta)[1]-1, :])#=}}}=#
    return yalpha, ybeta
end

function get_all_configs(config)
    #get all possible configs from a given start config
    all_configs = unique(collect(permutations(config, size(config)[1])))
    configs = []
    for i in 1:size(all_configs)[1]
        A = findall(!iszero, all_configs[i])
        push!(configs, A)
    end
    return configs
end

function get_index(config, y, orbs)
    #config has to be orbital index form since it will turned into bit string form{{{
    string = zeros(orbs)
    string[config] .= 1
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
    println("index:", Int8(index))
    return Int8(index)
end

function get_H_same(configs, nelec, norbs, y_matrix, int1e, int2e)
    #neleczo                                             is number of same spin electrons so alpha or beta{{{
    #norbs are number of orbitals
    #configs = list of all same spin configs of form: [1,4,5] orbital indexing for occupied orbs
    ndets = Int8(factorial(norbs) / (factorial(nelec)*factorial(norbs-nelec)))
    
    H_spin = zeros(ndets, ndets)
    I_idx = 0
    orbs = [1:norbs;]

    #loop through configs
    for I in configs
        I_idx = get_index(I, y_matrix, norbs)
        vir = filter!(x->!(x in I), orbs)

        #diagonal terms
        println("diagonal in H")
        H_spin[I_idx, I_idx] += 1*one_elec(I, int1e)
        H_spin[I_idx, I_idx] += 1*two_elec(I, int2e)
        
        #single excitations
        for i in I
            for a in vir
                config_single, sign_s = next_config(i, a, I)
                config_single_idx = get_index(config_single, y_matrix, norbs)
                println("single")
                H_spin[config_single_idx, I_idx] += sign_s*one_elec(config_single, int1e, i,a) 
                H_spin[config_single_idx, I_idx] += sign_s*two_elec(config_single, int2e, i,a)
                
                #double excitations
                for j in I
                    if j != i
                        for b in vir
                            if b != a
                                config_double, sign_d = next_config(j, b, config_single)
                                config_double_idx = get_index(config_double, y_matrix, norbs)
                                println("double")
                                H_spin[config_double_idx, I_idx] += sign_d*sign_s*two_elec(config_double, int2e, i, a, j, b)
                            end
                        end
                    end
                end
            end
        end
    end#=}}}=#
    return H_spin
end


function next_config(config, positions)
    #apply creation operator to the string{{{
    #get new index of string and store in lookup table
    #config is orbital indexing in ascending order
    #positions [i,a] meaning electron in orb_i went to orb_a
    spot = first(findall(x->x==positions[1], config))
    config[spot] = positions[2]
    sign = -1#=}}}=#
    return config, sign
end

function one_elec(config, int1e, i=0, a=0)
    h = 0 #={{{=#
    if i == 0
        #sum over all electrons (diag term)
        for m in config
            h += int1e[m, m]
        end
    else
        #only get single excitation contributation
        h += int1e[i,a]
    end#=}}}=#
    return h
end

function two_elec(config, int2e, i=0, a=0, j=0, b=0)
    #Anti symmetry included! compute the two electron contribution{{{
    g = 0
    println("\n", i,a,j,b)
    if i == 0 && a == 0 && j == 0 && b == 0 
        #sum over all electrons (diag term)
        println("diagonal in two e")
        for n in config
            for m in config
                if m > n
                    g += int2e[n,n,m,m]
                    g -= int2e[n,m,n,m]
                end
            end
        end

    if i!= 0 && a!=0 && j==0
        #single excitation 
        println("single")
        for m in config
            g += int2e[m,m,i,a]
            g -= int2e[m,i,m,a]
        end

    if i!= 0 && a!=0 && j!=0 && b!=0
        #double excitation
        println("double in two elec")
        g += int2e[i, j, a, b]
        g -= int2e[i, a, j, b]#=}}}=#
    println("g", g)
    end
    return g
end
end


function get_H_mixed(configs, nelec, norbs, y_matrix, int2e)
    #configs = [alpha_configs, beta_configs]{{{
    #nelec = [n_alpha, n_beta]
    #y_matrix = [y_alpha, y_beta]
    orbs = [1:norbs]
    ndets_a = factorial(norbs) / (factorial(nelec[1])*factorial(norbs-nelec[1])) 
    ndets_b = factorial(norbs) / (factorial(nelec[2])*factorial(norbs-nelec[2])) 
    sigma = zeros(ndets_a*ndets_b, ndets_a*ndets_b)

    for I in configs[1]  #alpha configs
        vir_I = filter!(x->!(x in I), orbs)
        for J in configs[2]     #beta configs
            vir_J = filter!(x->!(x in J), orbs)
            for i in I
                for a in vir_I
                    for j in J
                        for b in vir_J
                            #i think ill have some double counting in these
                            I_prim, signi = next_config(i, a, I)
                            J_prim, signj = next_config(j, b, J)
                            I_prim_idx = get_index(I_prim, y_matrix[1])
                            J_prim_idx = get_index(J_prim, y_matrix[2])
                            sigma[I_prim_idx, J_prim_idx] += int2e(i, j, a, b)
                        end
                    end
                end
            end
        end
    end#=}}}=#
    return sigma
end

function get_sigma(Ha, Hb, Hmixed,  vector)
    sigma1 = Ha*reshape(vector, (size(Ha)[1], size(Hb)[1]))
    sigma2 = Hb*reshape(vector, (size(Hb)[1], size(Ha)[1]))
    sigma3 = H_mixed*vector
    sigma = sigma1 + sigma2 + sigma3
    return sigma
end


end

