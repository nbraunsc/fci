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
        yalpha[:,1] .= 0
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
        #set first column to zeros for vert graph
        yalpha[:,1] .= 0
        ybeta[:,1] .= 0
        return (xalpha, xbeta), (yalpha, ybeta)
    end
end

function make_vert_graph(x)
    vert = Array{Int16}(zeros(size(x)))
    count = 1
    for row in 1:size(x)[1]
        for column in 1:size(x)[2]
            if x[row,column] != 0
                vert[row,column] = count
                count += 1
            end
        end
    end
    println("max: ", max)
    return vert
end

function make_ras_dict(y,vert)
    graph = Dict()
    for row in 1:size(y)[1]
        for column in 1:size(y)[2]
            #at last row and column
            if row==size(y)[1] && column==size(y)[2]
                return graph
            
            #at non existent node (RAS graphs)
            elseif vert[row,column] == 0
                println("at node zero")
                continue
            
            #at last row or no node present (RAS graphs)
            elseif row == size(y)[1] || vert[row+1,column]==0
                graph[vert[row,column]]=Set([vert[row,column+1]])
                #check if moving right is a node (RAS graphs)
                if vert[row,column+1] != 0
                    graph[vert[row,column],vert[row,column+1]] = y[row,column+1]
                end

            #at last column or no node present (RAS graphs)
            elseif column == size(y)[2] || vert[row,column+1]==0
                graph[vert[row,column]]=Set([vert[row+1, column]])
            

            else
                graph[vert[row,column]]=Set([vert[row,column+1],vert[row+1,column]])
                #check if moving right is a node (RAS graphs)
                if vert[row,column+1] != 0
                    graph[vert[row,column],vert[row,column+1]] = y[row,column+1]
                end
            end
        end
    end
    #max = findmax(ras_vert)[1]
    #println("max: ", max)
    return graph
end

function dfs_ras(graph, start, max, visited=Vector(zeros(max)), path=[], nodes=Dict())
    visited[start] = true
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index, config = get_index(path, graph)
        nodes[index] = config
    else
        for i in graph[start]
            if visited[i]==false
                dfs_ras(graph,i,max,visited,path,nodes)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return nodes
end

function get_index(path, graph)
    index = Int(1)
    config = Array{Int}(zeros(3))
    count = 1
    for i in 1:length(path)-1
        if (path[i],path[i+1]) in keys(graph)
            index += graph[(path[i],path[i+1])]
            config[count]=i
            count += 1
        end
    end
    return index, config
end


    
