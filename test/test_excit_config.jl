using fci
using JLD2

config = [1,2,3]
sign = 1

count = 2 #from bubble sort
arr = [2,3,4] #from bubble sort
positions = [1,4]

@testset "excit config" begin
    spot = first(findall(x->x==positions[1], config))
    config[spot] = positions[2]
    config_org = deepcopy(config) 
    count_test, arr_test = bubble_sort(config)
    @test count_test == count
    @test arr_test == arr
    if iseven(count_test)
        sign_test = 1
    else
        sign_test = -1
    end
    @test sign == sign_test
end
    
