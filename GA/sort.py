import random

# Testing function
def main():
    pvar_array = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    ivar_array = list(range(len(pvar_array)))
    print("Original pvar_array: ", pvar_array)
    print("Original ivar_array:", ivar_array)
    median_three_quick_sort(pvar_array, ivar_array)

    print("Sorted pvar_array:", pvar_array)
    print("Reordered ivar_array:", ivar_array)
    

# Quick sort implementation that sorts first array and puts second array in the same sequence
def median_three_quick_sort(pvar_array, ivar_array):
    def sort_helper(left, right):
        if right <= left:
            return
        a = random.randint(left, right)
        b = random.randint(left, right)
        c = random.randint(left, right)
        if pvar_array[a] <= pvar_array[b] <= pvar_array[c]:
            pivot_index = b
        elif pvar_array[b] <= pvar_array[a] <= pvar_array[c]:
            pivot_index = a
        else:
            pivot_index = c
        pivot_value = pvar_array[pivot_index]
        i, j = left, right
        while True:
            while pvar_array[i] < pivot_value:
                i += 1
            while pivot_value < pvar_array[j]:
                j -= 1
            if i >= j:
                break
            pvar_array[i], pvar_array[j] = pvar_array[j], pvar_array[i]
            ivar_array[i], ivar_array[j] = ivar_array[j], ivar_array[i]
            i += 1
            j -= 1
        sort_helper(left, j)
        sort_helper(j + 1, right)

    sort_helper(0, len(pvar_array) - 1)


if __name__ == "__main__":
    main()
