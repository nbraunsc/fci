using NPZ
using LinearAlgebra
using StaticArrays
#using fci

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
    max_val = findmax(vert)[1]
    #println("max: ", max)
    return vert, max_val
end

function make_graph_dict(y,vert)
    graph = Dict{Any, Any}()
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
                graph[vert[row,column]]=[vert[row,column+1]]
                #graph[vert[row,column]]=Set([vert[row,column+1]])
                #check if moving right is a node (RAS graphs)
                if vert[row,column+1] != 0
                    graph[vert[row,column],vert[row,column+1]] = y[row,column+1]
                end

            #at last column or no node present (RAS graphs)
            elseif column == size(y)[2] || vert[row,column+1]==0
                graph[vert[row,column]]=[vert[row+1, column]]
                #graph[vert[row,column]]=Set([vert[row+1, column]])
            

            else
                graph[vert[row,column]]=[vert[row,column+1],vert[row+1,column]]
                #graph[vert[row,column]]=Set([vert[row,column+1],vert[row+1,column]])
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

# Returns a node dictionary where keys are configs and values are the indexes
function dfs(nelecs, graph, start, max, visited=Vector(zeros(max)), path=[], nodes=Dict{Vector{Int32}, Int64}())
    visited[start] = true
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        index, config = get_index(nelecs, path, graph)
        #nodes[index] = config
        nodes[config] = index
    else
        for i in graph[start]
            if visited[i]==false
                dfs(nelecs, graph,i,max,visited,path,nodes)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return nodes
end

function get_index(nelecs, path, graph)
    index = 1 
    config = Vector{Int32}(zeros(nelecs))
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

            




