using NPZ
using LinearAlgebra
using StaticArrays
#using fci

#x = Array{Int}(zeros(4,4))
#x[1,:] .= [1,1,1,1]
#x[2,:] = [1,2,3,4]
#x[3,:] = [1,3,6,10]
#x[4,:] = [1,4,10,20]
#
#vert = Array{Int}(zeros(4,4))
#vert[1,:] = [1,2,3,4]
#vert[2,:] = [5,6,7,8]
#vert[3,:] = [9,10,11,12]
#vert[4,:] = [13,14,15,16]
#
#y = Array{Int}(zeros(4,4))
#y[1,:] .= [0,0,0,0]
#y[2,:] = [0,1,1,1]
#y[3,:] = [0,2,3,4]
#y[4,:] = [0,3,6,10]
##max = 16
#
##RAS Graphs for testing
#rasx = Array{Int}(zeros(4,4))
#rasx[1,:] = [1,1,1,0]
#rasx[2,:] = [1,2,3,3]
#rasx[3,:] = [0,0,3,6]
#rasx[4,:] = [0,0,3,9]
#
#rasy = Array{Int}(zeros(4,4))
#rasy[1,:] = [0,0,0,0]
#rasy[2,:] = [0,1,1,0]
#rasy[3,:] = [0,0,0,3]
#rasy[4,:] = [0,0,0,6]
#
#ras_vert = Array{Int}(zeros(4,4))
#ras_vert[1,:] = [1,2,3,0]
#ras_vert[2,:] = [4,5,6,7]
#ras_vert[3,:] = [0,0,8,9]
#ras_vert[4,:] = [0,0,10,11]

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
    max = findmax(vert)[1]
    #println("max: ", max)
    return vert, max
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

            




