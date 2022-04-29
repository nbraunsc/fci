using LinearAlgebra
using Printf

struct FCIProblem
    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    dima::Int 
    dimb::Int 
    dim::Int
    algorithm::String   #  options: lanczos/davidson
    n_roots::Int
end

function FCIProblem(no, na, nb)
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    dima = calc_ndets(no,na)
    dimb = calc_ndets(no,nb)
    return FCIProblem(no, na, nb, dima, dimb, dima*dimb, "lanczos", 1)
end

function display(p::FCIProblem)
    @printf(" FCIProblem:: #Orbs = %-3i #α = %-2i #β = %-2i Dimension: %-9i\n",p.no,p.na,p.nb,p.dim)
    #@printf(" FCIProblem::  NOrbs: %2i NAlpha: %2i NBeta: %2i Dimension: %-9i\n",p.no,p.na,p.nb,p.dim)
end

function calc_ndets(no,nelec)
    return factorial(no)÷(factorial(nelec)*factorial(no-nelec))
end


