file1='pide_hit.txt'
file2='deep_hit.txt'

pide={}
deep={}
with open(file1) as f1:
    f1.readline()
    for cnt, l in enumerate(f1):
        ls = l.strip().split(" ")
        pide[int(ls[0])-1]=int(ls[1])

with open(file2) as f2:
    f2.readline()
    for cnt, l in enumerate(f2):
        ls = l.strip().split(" ")
        deep[int(ls[0])]=int(ls[1])

max={}
count=0
for i in pide:
    if i in deep:
        if pide[i]>deep[i]:
            count+=1
            max[i]=[pide[i],deep[i]]
print(count)