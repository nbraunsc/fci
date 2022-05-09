using fci
using BenchmarkTools

function profile_fci(ints::H, p::FCIProblem, ci_vector=nothing, max_iter=12, nroots=1, tol=1e-6, precompute_ss=false)
    #This is the fci with lanczos that does not precompute same spin blocks
    fci.run_fci(ints, p, ci_vector, max_iter, nroots, tol, precompute_ss)
end

function profile_sigma3(a_configs, b_configs, a_lookup, b_lookup, ints::H, p::FCIProblem, ci_vector=nothing)
    @time sigma3 = vec(fci.compute_sigma_three(a_configs, b_configs, a_lookup, b_lookup, ci_vector, ints, p))
end

function profile_get_configs(norbs, nelecs)
    x, y = fci.make_xy(norbs, nelecs)
    fci.get_all_configs(x, y, nelecs)
end

function timing(p, ints, ci_vector, a_configs, b_configs, a_lookup, b_lookup)
    println("\n Time for computing all configs from depth first search")
    @btime conf = fci.compute_configs($p)[1]
    println("\n Time for filling lookup table")
    @btime a_look = fci.fill_lookup($p, $a_configs, $p.dima)
    #@btime a_look = fci.fill_lookup($p, $a_configs, $p.dima)
    println("\n Time for sigma 1 or 2")
    @btime sigma_two = fci.compute_sigma_two($a_configs, $a_lookup, $ci_vector, $ints, $p)
    println("\n Time for sigma 3")
    @btime sigma3 = vec(fci.compute_sigma_three($a_configs, $b_configs, $a_lookup, $b_lookup, $ci_vector, $ints, $p))
    println("\nTime for mat vec (all sigmas)")
    @btime sig = fci.matvec($a_configs, $b_configs, $a_lookup, $b_lookup, $ints, $p, $ci_vector)
end
