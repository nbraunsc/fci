using LinearAlgebra
using Printf
using fci
using StaticArrays

struct RASProblem
    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    fock::SVector{3, Int}   #fock section working in (ras1, ras2, ras3)
    ras1_min::Int       #min electrons in ras1
    ras3_max::Int       #max electrons in ras3
    algorithm::String   #  options: lanczos/davidson
    n_roots::Int
    dima::Int 
    dimb::Int 
    dim::Int
    xalpha::Array{Int}
    xbeta::Array{Int}
end

function RASProblem(no, na, nb, fock::Any, ras1_min=1, ras3_max=2)
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    fock = convert(SVector{3,Int},collect(fock))
    dima, xalpha = ras_calc_ndets(no, na, fock, ras1_min, ras3_max)
    dimb, xbeta = ras_calc_ndets(no, nb, fock, ras1_min, ras3_max)
    return RASProblem(no, na, nb, fock, ras1_min, ras3_max, "lanczos", 1, dima, dimb, dima*dimb, xalpha, xbeta)
end

function display(p::RASProblem)
    @printf(" FCIProblem:: #Orbs = %-3i #α = %-2i #β = %-2i Dimension: %-9i\n",p.no,p.na,p.nb,p.dim)
    #@printf(" FCIProblem::  NOrbs: %2i NAlpha: %2i NBeta: %2i Dimension: %-9i\n",p.no,p.na,p.nb,p.dim)
end

function ras_calc_ndets(no, nelec, fock, ras1_min, ras3_max)
    x = fci.make_ras_x(no, nelec, fock, ras1_min, ras3_max)
    dim_x = findmax(x)[1]
    return dim_x, x
end
