using fci
using JLD2
using LinearAlgebra
#using TimerOutputs

@load "_testdata_h8_integrals.jld2"
@load "_testdata_h8_alpha.jld2"
y_matrix = [yalpha, ybeta]

@load "_testdata_h8.jld2"
configs = [alpha_configs, beta_configs]

norbs = 8
nalpha = 4
ndets_a = factorial(norbs)÷(factorial(nalpha)*factorial(norbs-nalpha))
ndets_b = factorial(norbs)÷(factorial(4)*factorial(norbs-4))
dim = ndets_a*ndets_b

Ha = H_alpha
Hb = H_beta

Ia = Matrix{Float64}(I, size(Ha))
Ib = Matrix{Float64}(I, size(Hb))

#to = TimerOutput()

#@timeit to "test opt get sigma" begin
@time begin
@testset "get sigma" begin
    sigma_test1 = kron(Ib, Ha)*vector
    sigma_test2 = kron(Hb, Ia)*vector
    sigma_test3 = fci.opt_get_sigma3(configs, norbs, y_matrix, int2e, vector, index_table_a, index_table_b, dim, ndets_a)
    @test isapprox(sigma3, sigma_test3, atol=0.05)
    sigma_test = sigma_test1 + sigma_test2 + sigma_test3
    @test isapprox(sigma, sigma_test, atol=0.05)
end
end
