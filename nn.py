
import os
import sys
from fractions import Fraction 



def storyOfATree(n, edges, p, guesses,g):
    # print(guesses)
    if(p>g):
        return "0/1"
    adj_list = [[]for j in range(n)]
    for j in edges:
        adj_list[j[0]-1].append(j[1]-1)
        adj_list[j[1]-1].append(j[0]-1)
    counter = 0
    k = 0
    print(k)
    if(k == 0):
        li = [-1 for j in range(n)]
        tr = [k]
        li[k] = -2
        while(len(tr)>0):
            rr = tr[0]
            for j in adj_list[rr]:
                if(li[j] == -1):
                    tr.append(j)
                    li[j] = rr
            del tr[0]
    gue_list = [[] for k in range(n)]
    for gue in guesses:
        gue_list[gue[0]-1].append(gue[1]-1)
    count = 0
    for gue in guesses:
        if(li[gue[1]-1] == gue[0]-1):
            count = count + 1
    # print(gue_list)
    arr = [count for k in range(n)]
    pr = [0 for k in range(n)]
    pr[0] = 1
    tr = [0]
    while(len(tr)>0):
        rr = tr[0]
        print(rr)
        for j in adj_list[rr]:
            if(pr[j] == 0):
                tr.append(j)
                if j in gue_list[rr]:
                    arr[j] = arr[rr] -1
                if rr in gue_list[j]:
                    arr[j] = arr[rr] + 1
                pr[j] = 1  
        del tr[0]
        if(arr[rr]>=p):
            counter = counter + 1
    if(counter == 0):
        return "0/1"
    if(counter == n):
        return "1/1"
    return Fraction(counter,n)


if __name__ == '__main__':
    # fptr = open(os.environ['OUTPUT_PATH'], 'w')

    q = int(input())

    for q_itr in range(q):
        n = int(input())

        edges = []

        for _ in range(n-1):
            edges.append(list(map(int, input().rstrip().split())))

        gk = input().split()

        g = int(gk[0])

        k = int(gk[1])

        guesses = []

        for _ in range(g):
            guesses.append(list(map(int, input().rstrip().split())))
        result = storyOfATree(n, edges, k, guesses,g)
        print(result)
        # fptr.write(str(result) + '\n')

    # fptr.close()
