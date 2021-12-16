
function bubbleSort(arr)
    n = size(arr)[1]
    count = 0
 
    # Traverse through all array elements
    for i in 1:n+1
    #for i in 1:n
 
        # Last i elements are already in place
        for j in 1:n-i-1
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j+1] 
                count += 1
                arr[j], arr[j+1] = arr[j+1], arr[j]
            end
        end
    end
    return arr, count
end

arr = [64, 34, 25, 12, 22, 11, 90]

arr, count = bubbleSort(arr)
print(arr)
print(count)
