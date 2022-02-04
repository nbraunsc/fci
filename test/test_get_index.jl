using fci
using JLD2
using BenchmarkTools

@load "_testdata_h4_triplet_ymatrix.jld2"

norbs = 4      #for the h4 example
configs = [1,3,4], [4,2,3], [2], [1]
         #alpha, alpha, beta, beta
idxs = [3,4,2,1]

@testset "get index alpha" begin
    for i in 1:2
        idxb = fci.old_get_index(configs[i], ya, norbs)
        @btime fci.old_get_index($configs[$i], $ya, $norbs)
        idxa = fci.get_index(configs[i], ya, norbs)
        @btime fci.get_index($configs[$i], $ya, $norbs)
        @test idxs[i] == idxa
        @test idxs[i] == idxb
    end
end

#@testset "get index beta" begin
#    for i in 3:4
#        idxb = fci.old_get_index(configs[i], yb, norbs)
#        @time fci.old_get_index(configs[i], yb, norbs)
#        idxa = fci.get_index(configs[i], yb, norbs)
#        @time fci.get_index(configs[i], yb, norbs)
#        @test idxs[i] == idxa
#        @test idxs[i] == idxb
#    end
#end

