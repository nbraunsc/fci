using fci
using JLD2

@load "_testdata_h8.jld2"
@load "_testdata_h8_signs.jld2"
norbs = 8
nalpha = 4
ndets = factorial(norbs)÷(factorial(nalpha)*factorial(norbs-nalpha))
configs = alpha_configs
y_matrix = yalpha

@testset "make lookup tables" begin
    index1 = fci.old_make_index_table(configs, ndets, y_matrix)
    @time fci.old_make_index_table(configs, ndets, y_matrix)
    @test index1 == index_table_a

    index2, sign2 = fci.make_index_table(configs, ndets, y_matrix)
    @time fci.make_index_table(configs, ndets, y_matrix)
    @test index2 == index_table_a
    @test sign2 == sign_table_a
end


