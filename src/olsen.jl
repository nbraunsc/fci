using Pycall

function make_xy(norbs, nalpha, nbeta)
    #makes y matrices for grms indexing{{{
    n_unocc_a = (norbs-nalpha)+1
    n_unocc_b = (norbs-nbeta)+1

    #make x matricies
    xalpha = zeros(Float64, (n_unocc_a, nalpha+1))
    xbeta = zeros(Float64, (n_unocc_b, nbeta+1))
    #fill first row and columns
    xalpha[:,1] = 1
    xbeta[:,1] = 1
    xalpha[1,:] = 1
    xbeta[1,:] = 1

    for i in 2:nalpha+1
        for j in 2:n_unocc_a
            xalpha[j][i] = xalpha[j-1][i] + xalpha[j][i-1]
        end
    end
    
    for i in 2:nbeta+1
        for j in 2:n_unocc_b
            xbeta[j][i] = xbeta[j-1][i] + xbeta[j][i-1]
        end
    end

    #make y matrices
    copya = deepcopy(xalpha)
    copyb = deepcopy(xbeta)
    arraya = ones(size(xalpha)[2])
    arrayb = ones(size(xbeta)[2])
    yalpha = vcat(transpose(arraya), xalpha[1:size(xalpha)[2], :])
    ybeta = vcat(transpose(arrayb), xbeta[1:size(xbeta)[2], :])#=}}}=#
    return yalpha, ybeta
end

function get_idx(config, y)
    #config has to be in bit string form or turned into bit string form{{{
    index = 1
    start = [1,1]

    for value in config
        if value == 0
            #move down but dont add to index
            start[1] = start[1]+1
        end
        if value == 1
            #move right and add value to index
            start[2] = start[2] + 1
            index += y[start[1]][start[2]]
        end
    end#=}}}=#
    return index
end

function apply_creation(string, positions):
    #apply creation operator to the string
    #get new index of string and store in lookup table
    return new_config
end

function get_sigma(same_sign)
    #do some stuff
end

function get_sigma(mixed_sign)
    #do some stuff for mixed sign
end

function diagonalize(solver=lanczos)
    #get eigenvalues from lanczos
end


