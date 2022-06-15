import json
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
            if flag == 'none':
                flag = 'c'
                continue
            else:
                raise Exception("Some tag were never closed!")
        if x == "<A>":
            if flag == 'none':
                flag = 'A'
                continue
            else:
                raise Exception("Some tag were never closed!")
        if x == "<b>" : 
            if flag == 'none':
                flag = 'b'
                continue
            else:
                raise Exception("Some tag were never closed!")
        if x == "<x0>" :
            if flag == 'none':
                flag = 'x0'
                continue
            else:
                raise Exception("Some tag were never closed!")
        if x == "<s0>":
            if flag == 'none':
                flag = 's0'
                continue
            else:
                raise Exception("Some tag were never closed!")
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

def readJSONFile(filepath):

    f=open(filepath)
    df=json.load(f)
    f.close()
    
    for item in df['input']:
        for key in item:
            if key == 'c':
                c=item[key]
            if key == 'A':
                A=item[key]
            if key == 'b':
                b=item[key]
            if key == 'x0':
                x0=item[key]
            if key == 's0':
                s0=item[key]
    
    return np.asarray(c,dtype=np.float32), np.asarray(A,dtype=np.float64),np.asarray(b,dtype=np.float32) ,np.asarray(x0,dtype=np.float32), np.asarray(s0,dtype=np.float32)


#c,A,b,x0,s0 = readInputFile("input.txt")
#print(np.reshape(A,(np.asarray(b).shape[0] ,np.asarray(c).shape[0])))
#print(x0)
