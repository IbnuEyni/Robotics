from collections import defaultdict

def carPooling(trips, capacity):
    """
    :type trips: List[List[int]]
    :type capacity: int
    :rtype: bool
    """
    des = [0]*1001

    for n, f, t in trips:
        des[f-1] += n
        des[t] -= n
    return des
print(carPooling([[9,0,1],[3,3,7]], 4))