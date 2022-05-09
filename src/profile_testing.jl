using fci

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


