using fci
using JLD2
#using TimerOutputs

@load "_testdata_h8_alpha.jld2"
norbs = 8
nalpha = 4
ndets = factorial(norbs)÷(factorial(nalpha)*factorial(norbs-nalpha))
configs = alpha_configs
y_matrix = yalpha

#to = TimerOutput()
#@timeit to "test opt lookup table" begin
@time begin
@testset "lookup table" begin
    index_table_test = zeros(Int, configs[1].norbs, configs[1].norbs, ndets)
    orbs = [1:configs[1].norbs;]
    for I in 1:ndets
        vir = filter!(x->!(x in configs[I].config), [1:configs[I].norbs;])
        for p in configs[I].config
            for q in vir
                new_config, sorted_config, sign_s = fci.opt_excit_config(deepcopy(configs[I].config), [p,q])
                idx = fci.opt_get_index(new_config, y_matrix, configs[I].norbs)
                index_table_test[p,q,configs[I].label]=sign_s*idx
            end
        end
    end
    
    @test isapprox(index_table_a, index_table_test, atol=0.05)

end
end

