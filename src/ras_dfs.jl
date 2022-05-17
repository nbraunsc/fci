using fci
using StaticArrays

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

function make_ras_x(no, nelec, fock::SVector{3, Int}, ras1_min=1, ras3_max=2)
    x = make_x(no, nelec)
    keep_rows = fock[1]+1-ras1_min
    columns = nelec-ras3_max
    for i in keep_rows+1:size(x)[1]
        x[i,1:columns] .= 0 
        for column in columns+1:size(x)[2]
            x[i,column] = x[i-1,column]+x[i,column-1]
        end
    end
    return x
end

function make_ras_y(x, no::Int, nelec::Int, fock::SVector{3,Int}, ras1_min, ras3_max)
    #make RAS y graph
    n_unocc = (no-nelec)+1
    keep_rows = fock[1]+1-ras1_min
    columns = nelec-ras3_max
    yalpha = vcat(transpose(zeros(Int32, nelec+1)), x[1:n_unocc-1, :])
    for i in keep_rows+1:size(x)[1]
        yalpha[i,1:columns+1] .=0
    end
    yalpha[:,1] .= 0
    return yalpha
end

function make_vert_graph_ras(x)
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
    max_val = findmax(vert)[1]
    return vert, max_val
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

function dfs_ras(nelecs, graph, start, max, visited=Vector(zeros(max)), path=[], nodes=Dict())
    visited[start] = true
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index, config = get_index_ras(nelecs, path, graph)
        nodes[config] = index
    else
        for i in graph[start]
            if visited[i]==false
                dfs_ras(nelecs,graph,i,max,visited,path,nodes)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return nodes
end

function get_index_ras(nelecs, path, graph)
    index = 1
    config = Vector{Int}(zeros(nelecs))
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


    
