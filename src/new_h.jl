function sigma1(configs, nelec, norbs, y_matrix, int1e, int2e)
    ndets = Int8(factorial(norbs) / (factorial(nelec)*factorial(norbs-nelec)))
    orbs = [1:norbs;]
    Ha = zeros(ndets, ndets)

    for I in configs
        I_idx = get_index(I, y_matrix, norbs)
        F = zeros(ndets)
        vir = filter!(x->!(x in I), orbs)
        for i in I
            for a in vir
                config_single, sign_s = next_config(deepcopy(I), [i,a])
                config_single_idx = get_index(config_single, y_matrix, norbs)
                F[config_single_idx] += sign_s*one_elec(config_single, int1e, i,a) 
                F[config_single_idx] += sign_s*two_elec(config_single, int2e, i,a)
                
                for j in I
                    if j != i
                        for b in vir
                            if b != a
                                config_double, sign_d = next_config(deepcopy(config_single), [j, b])
                                config_double_idx = get_index(config_double, y_matrix, norbs)
                                F[config_double_idx] += sign_d*sign_s*two_elec(config_double, int2e, i, a, j, b)
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



                

                
