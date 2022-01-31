using fci
using JLD2

#using TimerOutputs

@load "_testdata_h8_alpha.jld2"

norbs = 8      #for the h4 example
configs = [4, 6, 7, 8], [5, 6, 7, 4], [2, 5, 6, 4], [2, 3, 6, 1]
         #alpha, alpha, beta, beta
idxs = [69, 35, 14, 6]

#norbs = 4      #for the h4 example
#configs = [1,3,4], [4,2,3], [2], [1]
         #alpha, alpha, beta, beta
#idxs = [3,4,2,1]

#to = TimerOutput()
#@timeit to "test opt get index" begin
@time begin
@testset "get index" begin
    for i in 1:4
        config = configs[i]
        idx = idxs[i]
        if i <= 2
            y = yalpha
            string = zeros(Int8, norbs)
            string[config] .= 1

            index = 1
            start = [1,1]

            for value in string
                if value == 0
                    #move down but dont add to index
                    start[1] = start[1]+1
                end
                if value == 1
                    #move right and add value to index
                    start[2] = start[2] + 1
                    index += y[start[1], start[2]]
                end
            end
            @test idx == index

        else
            y = ybeta
            string = zeros(Int8, norbs)
            string[config] .= 1

            index = 1
            start = [1,1]

            for value in string
                if value == 0
                    #move down but dont add to index
                    start[1] = start[1]+1
                end
                if value == 1
                    #move right and add value to index
                    start[2] = start[2] + 1
                    index += y[start[1], start[2]]
                end
            end
            @test idx == index
        end
    end
end
end
