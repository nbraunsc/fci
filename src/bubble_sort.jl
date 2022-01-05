
#function bubbleSort(arr)
#    n = size(arr)[1]
#    count = 0
# 
#    # Traverse through all array elements
#    #for i in 1:n+1
#    for i in 1:n
# 
#        # Last i elements are already in place
#        for j in 1:n-i-1
# 
#            # traverse the array from 0 to n-i-1
#            # Swap if the element found is greater
#            # than the next element
#            if arr[j] > arr[j+1] 
#                count += 1
#                arr[j], arr[j+1] = arr[j+1], arr[j]
#            end
#        end
#    end
#    return arr, count
#end
#
##arr = [64, 34, 25, 12, 22, 11, 90]
#
##arr, count = bubbleSort(arr)
#print(arr)
#print(count)

function bubbleSort(list)
    numberOfSwaps = 0
    numberOfComparisons = 0
    len = length(list)
    for i = 1:len-1
        for j = 2:len
            numberOfComparisons += 1
            if list[j-1] > list[j]
                tmp = list[j-1]
                list[j-1] = list[j]
                list[j] = tmp
                numberOfSwaps += 1
            end
        end
    end
    println("Number of swaps: ", numberOfSwaps)
    println("Number of comparisons: ", numberOfComparisons)
end


# UNCOMMENT CODE BELOW TO PRINT UNSORTED LIST

for i = 1:length(listToSort)
    println(listToSort[i])
end

# CALL BUBBLE SORT AND BENCHMARK IT

println("Bubble Sort")
@time bubbleSort(listToSort)
writedlm("sortResults.txt", listToSort)

# UNCOMMENT CODE BELOW TO PRINT SORTED LIST
for i = 1:length(listToSort)
    println(listToSort[i])
end

