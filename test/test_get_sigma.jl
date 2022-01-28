using fci
using JLD2
using LinearAlgebra

@load "_testdata_h4_triplet_integrals.jld2"
@load "_testdata_h4_triplet_ymatrix.jld2"
y_matrix = [ya, yb]
int1e = one
int2e = two

@load "_testdata_h4_triplet.jld2"
configs = [alpha_configs, beta_configs]

norbs = 4
nalpha = 3
ndets_a = factorial(norbs)÷(factorial(nalpha)*factorial(norbs-nalpha))
ndets_b = factorial(norbs)÷(factorial(1)*factorial(norbs-1))
dim = ndets_a*ndets_b


Ia = Matrix{Float64}(I, size(Ha))
Ib = Matrix{Float64}(I, size(Hb))

@testset "get sigma" begin
    sigma_test1 = kron(Ib, Ha)*vector
    sigma_test2 = kron(Hb, Ia)*vector
    sigma_test3 = fci.get_sigma3(configs, norbs, y_matrix, int2e, vector, index_table_a, index_table_b, dim, ndets_a)
    @test isapprox(sigma3, sigma_test3, atol=0.05)
    sigma_test = sigma_test1 + sigma_test2 + sigma_test3
    @test isapprox(sigma, sigma_test, atol=0.05)
end
