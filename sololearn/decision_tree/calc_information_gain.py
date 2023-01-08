S = [int(x) for x in input().split()]
A = [int(x) for x in input().split()]
B = [int(x) for x in input().split()]


def H(a):
    positive = float(len([x for x in a if x == 1])) / len(a)
    return 2 * positive * (1 - positive)


gain = H(S) - (float(len(A)) / len(S) * H(A)) - (float(len(B)) / len(S) * H(B))

print(round(gain, 5))
