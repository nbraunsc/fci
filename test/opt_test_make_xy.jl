using fci
using JLD2
#using TimerOutputs

@load "_testdata_h8_alpha.jld2"

norbs = 8
nalpha = 4
nbeta = 4

#to = TimerOutput()

#@timeit to "test opt make xy" begin
@time begin
@testset "make xy for grms" begin
    n_unocc_a = (norbs-nalpha)+1
    n_unocc_b = (norbs-nbeta)+1

    #make x matricies
    xalpha = zeros(n_unocc_a, nalpha+1)
    xbeta = zeros(n_unocc_b, nbeta+1)
    #fill first row and columns
    xalpha[:,1] .= 1
    xbeta[:,1] .= 1
    xalpha[1,:] .= 1
    xbeta[1,:] .= 1
    
    for i in 2:nalpha+1
        for j in 2:n_unocc_a
            xalpha[j, i] = xalpha[j-1, i] + xalpha[j, i-1]
        end
    end

    for i in 2:nbeta+1
        for j in 2:n_unocc_b
            xbeta[j, i] = xbeta[j-1, i] + xbeta[j, i-1]
        end
    end

    #make y matrices
    copya = deepcopy(xalpha)
    copyb = deepcopy(xbeta)
    arraya = zeros(size(xalpha)[2])
    arrayb = zeros(size(xbeta)[2])
    yalpha_test = vcat(transpose(arraya), xalpha[1:size(xalpha)[1]-1, :])
    ybeta_test = vcat(transpose(arrayb), xbeta[1:size(xbeta)[1]-1, :])#=}}}=#
    @test isapprox(yalpha, yalpha_test, atol=0.05)
    @test isapprox(ybeta, ybeta_test, atol=0.05)
end
end
    
