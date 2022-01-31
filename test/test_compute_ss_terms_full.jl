using fci
using JLD2

@load "_testdata_h8_integrals.jld2"
@load "_testdata_h8_alpha.jld2"
configs = alpha_configs
y_matrix = yalpha
norbs = 8
nalpha = 4
ndets = factorial(norbs)÷(factorial(nalpha)*factorial(norbs-nalpha))
index_table = index_table_a

@time begin
@testset "ss terms" begin
    Ha_test = zeros(size(H_alpha))
    nelecs = size(configs[1].config)[1]
    
    for I in configs #Ia or Ib, configs=list of all possible determinants
        I_idx = fci.get_index(I.config, y_matrix, I.norbs)
        F = zeros(ndets)
        orbs = [1:I.norbs;]
        vir = filter!(x->!(x in I.config), orbs)
       
        #single excitation
        for k in I.config
            for l in vir
                #annihlate electron in orb k
                config_single, sorted_config, sign_s = fci.excit_config(deepcopy(I.config), [k,l])
                config_single_idx = index_table[k,l,I.label]
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
                                single, sorted_s, sign_s = fci.excit_config(deepcopy(I.config), [k,l])
                                double, sorted_d, sign_d = fci.excit_config(deepcopy(sorted_s), [i,j])
                                idx = fci.get_index(double, y_matrix, I.norbs)
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

            
        Ha_test[:,I_idx] .= F
    end

    @test isapprox(H_alpha, Ha_test, atol=0.05)
end
end




