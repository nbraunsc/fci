using fci
using JLD2
using BenchmarkTools

@load "data/_testdata_h8.jld2"
norbs = 8
nalpha = 4
config = falses(norbs)
config[1:nalpha].=true

config_old = zeros(UInt8, norbs)
config_old[1:nalpha].=1

ndets = factorial(norbs)÷(factorial(nalpha)*factorial(norbs-nalpha))

@testset "get all configs" begin
    configs1 = fci.old_get_all_configs(config_old, norbs)
    @time fci.old_get_all_configs(config, norbs)

    configs2 = fci.get_all_configs(config, norbs, yalpha, nalpha, ndets)
    #@btime fci.get_all_configs($config, $norbs, $yalpha, $nalpha)
    
    @btime fci.get_all_configs($config_old, $norbs, $yalpha, $nalpha, $ndets)
    for i in 1:length(configs2)
        @test configs2[i].config == alpha_configs[i].config
    end
end

