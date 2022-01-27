using fci
using JLD2

@load "_testdata_h4_triplet_alphaconfigs.jld2"
configs = alpha_configs
y_matrix = ya
orbs = 4
nalpha = 3
ndets = factorial(orbs)÷(factorial(nalpha)*factorial(orbs-nalpha))
@load "_testdata_h4_triplet_integrals.jld2"


@testset "spin diag terms" begin
    Hout_test = zeros(ndets, ndets)
    for K in 1:ndets
        config = configs[K].config
        idx = get_index(config, y_matrix, orbs)
        for i in 1:nelec
            Hout_test[idx,idx] += int1e[config[i], config[i]]
            for j in i+1:nelec
                Hout_test[idx,idx] += int2e[config[i], config[i], config[j], config[j]]
                Hout_test[idx,idx] -= int2e[config[i], config[j], config[i], config[j]]
            end
        end
    end
    @test isapprox(Ha_diag, Hout_test, atol=0.05)
end
    
