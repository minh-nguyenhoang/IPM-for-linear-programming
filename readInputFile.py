import numpy as np

def readInputFile(filepath):
    df=[]
    c=[]
    A=[]
    b=[]
    x0=[]
    s0=[]
    with open(filepath) as f:
        if f.readable():
            for line in f.readlines():
                line_strip = line.strip()
                data = line_strip.split(" ")
                df.append(data)
    flattenedList = [x for xs in df for x in xs]  ## Flatten the nested list
    flag = "none"
    for x in flattenedList:
        if x in ['</c>','</A>','</b>','</x0>','</s0>']:
            flag = 'none'
        if x == '<c>':
            flag = 'c'
            continue
        if x == "<A>":
            flag = 'A'
            continue
        if x == "<b>":
            flag = 'b'
            continue
        if x == "<x0>":
            flag = 'x0'
            continue
        if x == "<s0>":
            flag = 's0'
            continue
        if flag == 'c':
            c.append(x)
        if flag == 'A':
            A.append(x)
        if flag == 'b':
            b.append(x)
        if flag == 'x0':
            x0.append(x)
        if flag == 's0':
            s0.append(x)

    return np.asarray(c,dtype=np.float32), np.reshape(np.asarray(A,dtype=np.float64),(np.asarray(b).shape[0] ,np.asarray(c).shape[0])) ,np.asarray(b,dtype=np.float32) ,np.asarray(x0,dtype=np.float32), np.asarray(s0,dtype=np.float32)



#c,A,b,x0,s0 = readInputFile("input.txt")
#print(np.reshape(A,(np.asarray(b).shape[0] ,np.asarray(c).shape[0])))
#print(x0)