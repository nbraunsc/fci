using Pycall

function make_xy(norbs, nalpha, nbeta)
    #makes y matrices for grms indexing{{{
    n_unocc_a = (norbs-nalpha)+1
    n_unocc_b = (norbs-nbeta)+1

    #make x matricies
    xalpha = zeros(Float64, (n_unocc_a, nalpha+1))
    xbeta = zeros(Float64, (n_unocc_b, nbeta+1))
    #fill first row and columns
    xalpha[:,1] = 1
    xbeta[:,1] = 1
    xalpha[1,:] = 1
    xbeta[1,:] = 1

    for i in 2:nalpha+1
        for j in 2:n_unocc_a
            xalpha[j][i] = xalpha[j-1][i] + xalpha[j][i-1]
        end
    end
    
    for i in 2:nbeta+1
        for j in 2:n_unocc_b
            xbeta[j][i] = xbeta[j-1][i] + xbeta[j][i-1]
        end
    end

    #make y matrices
    copya = deepcopy(xalpha)
    copyb = deepcopy(xbeta)
    arraya = ones(size(xalpha)[2])
    arrayb = ones(size(xbeta)[2])
    yalpha = vcat(transpose(arraya), xalpha[1:size(xalpha)[2], :])
    ybeta = vcat(transpose(arrayb), xbeta[1:size(xbeta)[2], :])#=}}}=#
    return yalpha, ybeta
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
            index += y[start[1]][start[2]]
        end
    end#=}}}=#
    return index
end

function apply_creation(config, positions):
    #apply creation operator to the string{{{
    #get new index of string and store in lookup table
    #config is orbital indexing in ascending order
    #positions [i,a] meaning electron in orb_i went to orb_a
    spot = first(findall(x->x==positions[1], config))
    config[spot] = positions[2]
    sign = -1#=}}}=#
    return config, sign
end

function one_elec!(config, i=0, a=0)
    h = 0#={{{=#
    if i = 0
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

function two_elec!(config, i=0, a=0, j=0, b=0)
    #Anti symmetry included! compute the two electron contribution{{{
    g = 0
    if i = 0 and a = 0 and j=0 and b=0
        #sum over all electrons (diag term)
        for n in config
            for m > n in config
                g += int2e[n,n,m,m]
                g -= int2e[n,m,n,m]
            end
        end

    if i!= 0 and a!=0 and j=0
        #single excitation 
        for m in config
            g += int2e[m,m,i,a]
            g -= int2e[m,i,m,a]
        end

    else
        #double excitation
        g += int2e[i, j, a, b]
        g -= int2e[i, a, j, b]#=}}}=#
    return g
end

function get_sigma(configs, nelec, norbs, y_matrix, spin="same")
    #nelec is number of same spin electrons so alpha or beta{{{
    #norbs are number of orbitals
    #configs = list of all same spin configs of form: [1,4,5] orbital indexing for occupied orbs
    ndets = factorial(norbs) / (factorial(nelec)*factorial(norbs-nelec)) 
    H_spin = zeros(ndets, ndets)
    I_idx = 0
    orbs = [1:norbs]

    #loop through configs
    for I in configs
        I_idx = get_index(I, y_matrix)
        vir = filter!(x->!(x in I), orbs)

        #diagonal terms
        H_spin[I_idx, I_idx] += 1*one_elec(I)
        H_spin[I_idx, I_idx] += 1*two_elec(I)
        
        #single excitations
        for i in I
            for a in vir
                config_single, sign_s = apply_creation(i, a, I)
                config_single_idx = get_index(config_single, y_matrix)
                H_spin[config_single_idx, I_idx] += sign_s*one_elec(config_single, i,a) 
                H_spin[config_single_idx, I_idx] += sign_s*two_elec(config_single, i,a)
                
                #double excitations
                for j != i in I
                    for b != a in vir
                        config_double, sign_d = apply_creation(j, b, config_single)
                        config_double_idx = get_index(config_double, y_matrix)
                        H_spin[config_double_idx, I_idx] += sign_d*sign_s*two_elec(config_double, i, a, j, b)
                    end
                end
            end
        end
    end#=}}}=#
    sigma = H_spin*CI_vector
    return sigma
end

function get_sigma(configs, nelec, norbs, y_matrix, spin="mixed")
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
                            I_prim, signi = apply_creation(i, a, I)
                            J_prim, signj = apply_creation(j, b, J)
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

function diagonalize(solver=lanczos)
    #get eigenvalues from lanczos
end


