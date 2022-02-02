using fci
using JLD2
using Combinatorics
#using TimerOutputs

@load "_testdata_h8_alpha.jld2"
configs = alpha_configs

config = [1,1,1,1,0,0,0,0]
norbs = 8
nelecs = 4

#to = TimerOutput()
#@timeit to "test opt get all configs" begin
@time begin
@testset "get all configs" begin
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
end
#print_timer(to::TimerOutput)
