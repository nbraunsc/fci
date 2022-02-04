using StaticArrays

struct DeterminantString{N}
    norbs::Int16
    nelec::UInt8
    config::SVector{N, Int}
    #config::MVector{N, UInt}
    index::UInt
end
