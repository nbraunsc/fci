using fci

function make_vert_graph(x)
    vert = Array{Int}(zeros(size(x)))
    count = 1
    for row in 1:size(x)[1]
        for column in 1:size(x)[2]
            if x[row,column] != 0
                vert[row,column] = count
                count += 1
            end
        end
    end
    max_value = findmax(vert)[1]
    println("max: ", max_value)
    return vert, max_value
end

function make_graph_dict(y,vert)
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

function start_ras_graphs(norbs, nalpha, nbeta)
    xalpha, yalpha, xbeta, ybeta = fci.make_xy(norbs, nalpha, nbeta)
    if nalpha == nbeta
        #closed shell only need one set of sping graphs
        vert, max = make_vert_graph(xalpha)
        graph = make_graph_dict(yalpha, vert)
        return graph, max
    end
end

function make_rasorbs(rasi_orbs, rasiii_orbs, norbs, frozen_core=false)
    if frozen_core==false
        i_orbs = [1:1:rasi_orbs;]
        start = norbs-rasiii_orbs+1
        iii_orbs = [start:1:norbs;]
        return i_orbs, iii_orbs
    end
end

function check_min_rasi(config, i_orbs, rasi_min)
    #if at least 1 electron in rasi then return true else return false
    value = 0
    for num in i_orbs
        value += count(i->(i==num), config)
        if value >= rasi_min
            return true
        else
            continue
        end
    return false
    end
end

function calc_elecs_rasiii(config, iii_orbs, rasiii_max)
    #calculate number of electrons in rasiii and return value
    value = 0
    for num in iii_orbs
        value += count(i->(i==num), config)
    return value
    end
end

function ras_dfs(graph, start, max, rasi_min=1, rasiii_max=2, x, visited=Vector(zeros(max)), path=[], ras0e=Vector{Vector{Int}}(undef,findmax(x)[1]), ras1e=Vector{Vector{Int}}(undef,findmax(x)[1]), ras2e=Vector{Vector{Int}}(undef,findmax(x)[1]))
    visited[start] = true
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index, config = get_index(path, graph)
        if check_min_rasi(config, i_orbs, rasi_min) == true
            rasiii_elec = calc_elecs_rasiii(config, iii_orbs, rasiii_max)
            if rasiii_elec == 0
                ras0e[index] = config
            elseif rasiii_elec == 1
                ras1e[index] = config
            elseif rasiii_elec == 2
                ras2e[index] = config
            end
        end
    else
        for i in graph[start]
            if visited[i]==false
                ras_dfs(graph,i,max,rasi_min, rasiii_max, x, visited,path,ras0e,ras1e,ras2e)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return ras0e, ras1e, ras2e
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

