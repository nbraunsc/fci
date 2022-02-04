using StaticArrays

struct DeterminantString{N}
    norbs::Int
    nelec::Int
    config::SVector{N, Int}
    #config::MVector{N, UInt}
    index::Int
end
