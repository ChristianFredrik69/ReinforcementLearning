from collections import defaultdict
print("Hello World!")

dictor = defaultdict(lambda: [0.0] * 2)


dictor['Peter'][0] = 4.5
dictor['Peter'][1] = 9

print(dictor)

fred = ("bob", "petra", "petter")

print(fred[1])

print(int(100_000))
print(100_000)


def my_function():
    a = 10  # 'a' is local to this function
    if True:
        b = 5  # 'b' is also local to this function, not just to the 'if' block
    print(a, b)  # prints: 10 5

my_function()

def klubb():
    for i in range (10):
        b = i
    print(b)

def kløbb():
    i = 0
    while (i < 10):
        i += 1
        b = i

    print(b)

klubb()
kløbb()