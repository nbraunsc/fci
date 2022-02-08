using fci
using JLD2
using BenchmarkTools

@load "data/_testdata_h8.jld2"
norbs = 8
nalpha = 4
nbeta = 4

@testset "make xy for grms" begin
    ya1, yb1 = fci.old_make_xy(norbs, nalpha, nbeta)
    println("\nOld Make XY @time")
    @btime fci.old_make_xy($norbs, $nalpha, $nbeta)
    #@time fci.old_make_xy(norbs, nalpha, nbeta)
    #@test isapprox(ya1, yalpha, atol=10e-7)
    #@test isapprox(yb1, ybeta, atol=10e-7)
    
    ya2, yb2 = fci.make_xy(norbs, nalpha, nbeta)
    println("\nNew Make XY @time")
    @btime fci.make_xy($norbs, $nalpha, $nbeta)
    #@time fci.make_xy(norbs, nalpha, nbeta)
    #@test isapprox(ya2, yalpha, atol=10e-7)
    #@test isapprox(yb2, ybeta, atol=10e-7)
end
    
