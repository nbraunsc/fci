using fci
using StaticArrays

config = SVector{4,Int}([1, 3, 4, 5])
positions = [1,6]

sign = -1
count = 3 #from bubble sort
arr = [3, 4, 5, 6] #from bubble sort
i = 1
j = 6

@testset "excit config" begin
    config1, arr1, sign1 = fci.old_excit_config(config, positions)
    println("\n Old Excit Config")
    @time fci.old_excit_config(config, positions)
    @test sign == sign1
    #@test arr == arr1
    
    config2, arr2, sign2 = fci.excit_config(config, 1, 6)
    println("\n New Excit Config")
    @time fci.excit_config(config, 1, 6)
    @test sign == sign2
    #@test arr == arr2
end

