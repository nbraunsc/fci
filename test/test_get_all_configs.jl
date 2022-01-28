using fci
using JLD2
using Combinatorics

@load "_testdata_h4_triplet_alphaconfigs.jld2"
configs = alpha_configs

config = [1,1,1,0]
norbs = 4

@testset "get all configs" begin
    nelecs = UInt8(size(config)[1])
    all_configs_test = unique(collect(permutations(config, size(config)[1])))
    configs_test = [] #empty Array
    for i in 1:size(all_configs_test)[1]
        A = findall(!iszero, all_configs_test[i])
        push!(configs_test,fci.DeterminantString(norbs, nelecs, A, 1, UInt(i))) 
    end

    for i in 1:length(configs)
        @test configs[i].config == configs_test[i].config
    end
end

