
mutable struct DeterminantString
    norbs::UInt8
    nelec::UInt8
    config::Vector{Int}
    #config::MVector{N, Int}
    index::UInt
end
