using fci
using JLD2

@load "_testdata_h4_triplet_alphaconfigs.jld2"
norbs = 4
nalpha = 3
ndets = factorial(norbs)÷(factorial(nalpha)*factorial(norbs-nalpha))
configs = alpha_configs
y_matrix = ya

@testset "lookup table" begin
    index_table_test = zeros(Int, configs[1].norbs, configs[1].norbs, ndets)
    orbs = [1:configs[1].norbs;]
    for I in 1:ndets
        vir = filter!(x->!(x in configs[I].config), [1:configs[I].norbs;])
        for p in configs[I].config
            for q in vir
                new_config, sorted_config, sign_s = excit_config(deepcopy(configs[I].config), [p,q])
                idx = get_index(new_config, y_matrix, configs[I].norbs)
                index_table_test[p,q,configs[I].label]=sign_s*idx
            end
        end
    end
    
    @test isapprox(index_table_a, index_table_test, atol=0.05)

end

