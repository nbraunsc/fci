using fci

function make_x(norbs, nalpha)
    #makes y matrices for grms indexing{{{
    n_unocc_a = (norbs-nalpha)+1

    #make x matricies
    xalpha = zeros(Int, n_unocc_a, nalpha+1)
    #fill first row and columns
    xalpha[:,1] .= 1
    xalpha[1,:] .= 1
    
    for i in 2:nalpha+1
        for j in 2:n_unocc_a
            xalpha[j, i] = xalpha[j-1, i] + xalpha[j, i-1]
        end
    end
    return xalpha#=}}}=#
end

function make_ras_graphs(nalpha, nbeta, rasi_min=1, rasiii_max=2, fock=Vector(zeros(3)))
    norbs = fock[1] + fock[2] + fock[3]
    if nalpha == nbeta #i.e. closed shell
        n_unocc_a = (norbs-nalpha)+1#={{{=#
        xalpha = make_x(norbs, nalpha)
        keep_rows = fock[1]+1-rasi_min
        columns = nalpha-rasiii_max
        for i in keep_rows+1:size(xalpha)[1]
            xalpha[i,1:columns] .= 0 
            for column in columns+1:size(xalpha)[2]
                xalpha[i,column] = xalpha[i-1,column]+xalpha[i,column-1]
            end
        end
        #make RAS y graph
        yalpha = vcat(transpose(zeros(Int16, nalpha+1)), xalpha[1:n_unocc_a-1, :])
        for i in keep_rows+1:size(xalpha)[1]
            yalpha[i,1:columns+1] .=0
        end#=}}}=#
        return xalpha, yalpha
    
    else #not closed shell 
        xalpha = make_x(norbs, nalpha)#={{{=#
        xbeta = make_x(norbs, nbeta)
        n_unocc_a = (norbs-nalpha)+1
        n_unocc_b = (norbs-nbeta)+1
        keep_rows = fock[1]+1-rasi_min
        columns_a = nalpha-rasiii_max
        columns_b = nbeta-rasiii_max
        for i in keep_rows+1:size(xalpha)[1]
            xalpha[i,1:columns_a] .= 0 
            for column in columns_a+1:size(xalpha)[2]
                xalpha[i,column] = xalpha[i-1,column]+xalpha[i,column-1]
            end
        end
        #make RAS y graph for alpha
        yalpha = vcat(transpose(zeros(Int16, nalpha+1)), xalpha[1:n_unocc_a-1, :])
        for i in keep_rows+1:size(xalpha)[1]
            yalpha[i,1:columns_a+1] .=0
        end

        for j in keep_rows+1:size(xbeta)[1]
            xbeta[j,1:columns_b] .= 0 
            for column in columns_b+1:size(xbeta)[2]
                xbeta[j,column] = xbeta[j-1,column]+xbeta[j,column-1]
            end
        end
        #make RAS y graph for beta
        ybeta = vcat(transpose(zeros(Int16, nbeta+1)), xbeta[1:n_unocc_b-1, :])
        for i in keep_rows+1:size(xbeta)[1]
            ybeta[i,1:columns_b+1] .=0
        end#=}}}=#
        return (xalpha, xbeta), (yalpha, ybeta)
    end
end


    
