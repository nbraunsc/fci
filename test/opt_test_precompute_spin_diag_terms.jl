using fci
using JLD2
#using TimerOutputs

@load "_testdata_h8_alpha.jld2"
configs = alpha_configs
y_matrix = yalpha
orbs = 8
nalpha = 4
ndets = factorial(orbs)÷(factorial(nalpha)*factorial(orbs-nalpha))
@load "_testdata_h8_integrals.jld2"


#to = TimerOutput()

#@timeit to "test opt spin diag terms" begin
@time begin
@testset "spin diag terms" begin
    Hout_test = zeros(ndets, ndets)
    for K in 1:ndets
        config = configs[K].config
        idx = fci.opt_get_index(config, y_matrix, orbs)
        for i in 1:nalpha
            Hout_test[idx,idx] += int1e[config[i], config[i]]
            for j in i+1:nalpha
                Hout_test[idx,idx] += int2e[config[i], config[i], config[j], config[j]]
                Hout_test[idx,idx] -= int2e[config[i], config[j], config[i], config[j]]
            end
        end
    end
    @test isapprox(Ha_diag, Hout_test, atol=0.05)
end
end
    
