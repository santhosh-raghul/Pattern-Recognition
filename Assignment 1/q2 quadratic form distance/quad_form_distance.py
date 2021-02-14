import numpy as np

def matrix_mult(A,B):
    p=len(A)
    q=len(A[0])
    r=len(B)
    s=len(B[0])
    if(q!=r):
        return None
    pd=[]
    for i in range(p):
        row=[]
        for j in range(s):
            x=0
            for k in range(q):
                x+=A[i][k]*B[k][j]
            row.append(x)
        pd.append(row)
    return pd

def transpose(A):
    tr=[]
    for j in range(len(A[0])):
        row=[]
        for i in range(len(A)):
            row.append(A[i][j])
        tr.append(row)
    return tr

hq_ht_t = [[0.5,0.5,-0.5,-0.25,-0.25]] # transpose(hq_ht)
A=[
    [1,0.135,0.195,0.137,0.157],
    [0.135,1,0.2,0.309,0.143],
    [0.195,0.2,1,0.157,0.122],
    [0.137,0.309,0.157,1,0.195],
    [0.157,0.143,0.122,0.195,1]
]

# Distance = (Transpose(hq-ht))*A*(hq-ht)
dist=matrix_mult((matrix_mult(hq_ht_t,A)),(transpose(hq_ht_t)))
print("Quadratic form distance = %.6f"%dist[0][0])

print("\nVerification using inbuilt function:")
dist=np.dot((np.dot(hq_ht_t,A)),(np.transpose(hq_ht_t)))
print("Quadratic form distance = %.6f"%dist[0][0])