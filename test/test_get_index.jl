using fci
using JLD2

@load "_testdata_h4_triplet_ymatrix.jld2"

norbs = 4      #for the h4 example
configs = [1,3,4], [4,2,3], [2], [1]
         #alpha, alpha, beta, beta
idxs = [3,4,2,1]

@testset "get index" begin
    for i in 1:4
        config = configs[i]
        idx = idxs[i]
        if i <= 2
            y = ya
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
            y = yb
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
