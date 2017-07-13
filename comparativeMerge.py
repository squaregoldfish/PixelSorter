''' Module for comparative merge sorting '''

def merge(pri_items, sec_items, pri_aux, sec_aux, lowerbound, upperbound):
    ''' docstring '''
    mid = int(lowerbound + (upperbound - lowerbound)/2)
    i = lowerbound
    j = mid + 1
    for k in range(lowerbound, upperbound + 1):
        pri_aux[k] = pri_items[k]
        sec_aux[k] = sec_items[k]
    for index in range(lowerbound, upperbound + 1):
        if i > mid:
            pri_items[index] = pri_aux[j]
            sec_items[index] = sec_aux[j]
            j += 1
        elif j > upperbound:
            pri_items[index] = pri_aux[i]
            sec_items[index] = sec_aux[i]
            i += 1
        elif less(pri_aux[i], pri_aux[j]):
            pri_items[index] = pri_aux[j]
            sec_items[index] = sec_aux[j]
            j += 1
        else:
            pri_items[index] = pri_aux[i]
            sec_items[index] = sec_aux[i]
            i += 1

def merge_sort(pri_items, sec_items):
    ''' docstring '''
    pri_aux = []
    sec_aux = []
    for i in range(len(pri_items)):
        pri_aux.append(0)
        sec_aux.append(0)
    upperbound = len(pri_items) - 1
    lowerbound = 0
    sort(pri_items, sec_items, pri_aux, sec_aux, lowerbound, upperbound)
    assert is_sorted(pri_items)

def sort(pri_items, sec_items, pri_aux, sec_aux, lowerbound, upperbound):
    ''' docstring '''
    if upperbound <= lowerbound:
        return
    mid = int(lowerbound + (upperbound - lowerbound)/2)
    sort(pri_items, sec_items, pri_aux, sec_aux, lowerbound, mid)
    sort(pri_items, sec_items, pri_aux, sec_aux, mid + 1, upperbound)
    merge(pri_items, sec_items, pri_aux, sec_aux, lowerbound, upperbound)

def is_sorted(pri_items):
    ''' docstring '''
    for i in range(1, len(pri_items)):
        if pri_items[i] < pri_items[i-1]:
            return False
    return True

def less(value_one, value_two):
    ''' docstring '''
    return value_one > value_two