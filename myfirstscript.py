# problem #1
# What is mising in the while loop?
# use a breakpoint in line 6 to debug
x = 1
while x < 10:
    print(x)
    x = x + 1


# problem #2
# use a breakpoint in line 14 to debug
mylist = list(range(5))

for x in range(1, 6):
    print(f"Run No.:{mylist[x]}")
# here the index is out of range, so it only prints
# elements 2, 3, 4, 5 of `mylist`, which is elements
# 1, 2, 3, 4. basically, but for x, there is not a
# way to index mylist by the latter two elements of x
