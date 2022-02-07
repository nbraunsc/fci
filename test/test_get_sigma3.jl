using fci
using JLD2
using LinearAlgebra
using BenchmarkTools

@load "data/_testdata_h8_integrals.jld2"
@load "data/_testdata_h8.jld2"
@load "data/_testdata_h8_signs.jld2"
y_matrix = [yalpha, ybeta]
configs = [alpha_configs, beta_configs]

norbs = 8
nalpha = 4
nbeta = 4
ndets_a = factorial(norbs)÷(factorial(nalpha)*factorial(norbs-nalpha))
ndets_b = factorial(norbs)÷(factorial(nbeta)*factorial(norbs-nbeta))
dim = ndets_a*ndets_b


#Ia = Matrix{Float64}(I, size(Ha))
#Ib = Matrix{Float64}(I, size(Hb))

@testset "get sigma3" begin
    oldsigma_test3 = fci.old_get_sigma3(configs, norbs, y_matrix, int2e, vector, index_table_a, index_table_b, dim, ndets_a)
    println("\nOld Get Sigma3 @time, then @btime")
    @time fci.old_get_sigma3(configs, norbs, y_matrix, int2e, vector, index_table_a, index_table_b, dim, ndets_a)
    @btime fci.old_get_sigma3($configs, $norbs, $y_matrix, $int2e, $vector, $index_table_a, $index_table_b, $dim, $ndets_a)
    @test isapprox(sigma3, oldsigma_test3, atol=10e-7)
    
    sigma_test3 = fci.get_sigma3(configs, norbs, int2e, vector, index_table_a, index_table_b, dim, ndets_a, sign_table_a, sign_table_b)
    println("\nNew Get Sigma3 @time, then @btime")
    @time fci.get_sigma3(configs, norbs, int2e, vector, index_table_a, index_table_b, dim, ndets_a, sign_table_a, sign_table_b)
    @btime fci.get_sigma3($configs, $norbs, $int2e, $vector, $index_table_a, $index_table_b, $dim, $ndets_a, $sign_table_a, $sign_table_b)
    @test isapprox(sigma3, sigma_test3, atol=10e-7)
end
