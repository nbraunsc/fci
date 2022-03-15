using NPZ
using LinearAlgebra
using StaticArrays
using fci

x = Array(zeros(4,4))
x[1,:] .= [1,1,1,1]
x[2,:] = [1,2,3,4]
x[3,:] = [1,3,6,10]
x[4,:] = [1,4,10,20]

y = Array(zeros(4,4))
y[1,:] .= [0,0,0,0]
y[2,:] = [0,1,1,1]
y[3,:] = [0,2,3,4]
y[4,:] = [0,3,6,10]

function find_first_path(x,y)
    start = [1,1]
    index = 1
    config = []
    orb = 0
    #move all the way right
    for i in y[1,:]
        start[2] += 1 
        index += y[start[1],start[2]]
        orb += 1
        push!(config, orb)
    end

    #move all way down
    for j in y[:,start[2]]
        start[1] += 1
        orb += 1
        #check to see if there is right direction on path
        if not check_right(y, start, index, orb, config)
            continue
        end
    end
end

function move_right(y, start, index, orb, config)
    start[2] += 1
    index += y[start[1], start[2]]
    orb += 1
    push!(config, orb)
    return start, index, orb, config
end

function move_down(y, start, orb, config)
    start[1] += 1
    orb += 1
    return start, orb
end


function check_right(y, start, index, orb, config)
    if start[start[1],start[2]+1] > 0
        start[2] += 1
        index += y[start[1], start[2]]
        orb += 1
        push!(config, orb)
        if start[start[1],start[2]+1] > 0
            check_right(y,start,index,orb,config)
        else
            return y, start, index, orb, config
        end

    else
        return false
    end
end


        


