using fci
using JLD2
using LinearAlgebra
using BenchmarkTools

@load "data/_testdata_h8_integrals.jld2"
@load "data/_testdata_h8.jld2"
@load "data/_testdata_h8_signs.jld2"
Ha = H_alpha - Ha_diag

norbs = 8
nalpha = 4
nbeta = 4
ndets = factorial(norbs)÷(factorial(nalpha)*factorial(norbs-nalpha))

@testset "compute same spin" begin
    ha2 = fci.old_compute_ss_terms_full(ndets, norbs, int1e, int2e, index_table_a, alpha_configs, yalpha)
    println("\nOld Compute same spin @time, then @btime")
    @time fci.old_compute_ss_terms_full(ndets, norbs, int1e, int2e, index_table_a, alpha_configs, yalpha)
    @btime fci.old_compute_ss_terms_full($ndets, $norbs, $int1e, $int2e, $index_table_a, $alpha_configs, $yalpha)
    @test isapprox(Ha, ha2, atol=10e-7)
    
    ha1 = fci.compute_ss_terms_full(ndets, norbs, int1e, int2e, index_table_a, alpha_configs, yalpha, sign_table_a)
    println("\nNew Compute same spin @time, then @btime")
    @time fci.compute_ss_terms_full(ndets, norbs, int1e, int2e, index_table_a, alpha_configs, yalpha, sign_table_a)
    @btime fci.compute_ss_terms_full($ndets, $norbs, $int1e, $int2e, $index_table_a, $alpha_configs, $yalpha, $sign_table_a)
    @test isapprox(Ha, ha1, atol=10e-7)
 
end

