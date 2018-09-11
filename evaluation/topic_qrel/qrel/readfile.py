

with open("output1.txt") as f:
    tokens = f.readlines()
    for line in tokens:
        line = line.split()
        if line[-1] == '1':
            print (" ".join(line))