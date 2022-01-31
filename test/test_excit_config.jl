using fci
using JLD2

#config = [1,2,3,4]
#sign = 1

#count = 2 #from bubble sort
#arr = [2,3,4] #from bubble sort
#positions = [1,4]

config = [1, 3, 4, 5]
sign = -1

count = 3 #from bubble sort
arr = [3, 4, 5, 6] #from bubble sort
positions = [1,6]
@time begin
@testset "excit config" begin
    spot = first(findall(x->x==positions[1], config))
    config[spot] = positions[2]
    config_org = deepcopy(config) 
    count_test, arr_test = fci.bubble_sort(config)
    @test count_test == count
    @test arr_test == arr
    if iseven(count_test)
        sign_test = 1
    else
        sign_test = -1
    end
    @test sign == sign_test
end
end    
