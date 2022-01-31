using fci
using Test

@testset "fci.jl" begin
    include("opt_test_compute_ss_terms_full.jl")
    include("test_compute_ss_terms_full.jl")

    include("opt_test_precompute_spin_diag_terms.jl")
    include("test_precompute_spin_diag_terms.jl")

    include("opt_test_excit_config.jl")
    include("test_excit_config.jl")

    include("opt_test_get_all_configs.jl")
    include("test_get_all_configs.jl")

    include("opt_test_get_index.jl")
    include("test_get_index.jl")

    include("opt_test_get_sigma.jl")
    include("test_get_sigma.jl")

    include("opt_test_make_lookup_table.jl")
    include("test_make_lookup_table.jl")

    include("opt_test_make_xy.jl")
    include("test_make_xy.jl")
end
