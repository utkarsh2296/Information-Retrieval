import function

w1 = input("enter query word 1: >")
w2 = input("enter query word 2: >")
list1 = function.preprocess(w1)
list2 = function.preprocess(w2)

print(len(list1))
print(len(list2))

print("Select the operation :")
print("1. And")
print("2. or")
print("3. orNot")
print("4. AndNot")
print("5. skip method for and")

inp = int(input(">"))

if(inp == 1):
    r = function.andFunction(list1, list2)
    print(r)
    print("Length is :", len(r))
elif inp == 2:
    r = (function.orFunction(list1, list2))
    print("\nLength is :", len(r))
elif inp == 3:
    r = (function.orNot(list1, list2))
    print(r, "\nLength is :", len(r))
elif inp == 4:
    r = (function.andNot(list1, list2))
    print(r, "\nLength is :", len(r))
elif inp == 5:
    r, _, _ = (function.skipPointer(list1, list2))
    print(r, "\nLength is :", len(r))
else:
    print("wrong choice!")
