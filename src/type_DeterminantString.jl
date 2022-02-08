using StaticArrays

struct DeterminantString{N}
    norbs::UInt8
    nelec::UInt8
    config::SVector{N, UInt8}
    #config::MVector{N, UInt}
    index::UInt
end

function DeterminantString(norbs, nelec, in::AbstractArray{T,1}, index) where T 
    return DeterminantString{length(in)}(convert(UInt8, norbs), 
                                  convert(UInt8, nelec), 
                                  ntuple(i -> convert(UInt8, in[i]), length(in)),
                                  convert(UInt, index)
                                 )
end

"""
    function Base.convert(
"""

