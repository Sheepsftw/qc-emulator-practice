import numpy as np
import math

# classical gates
I = np.array([[1, 0], [0, 1]])
NOT = np.array([[0, 1], [1, 0]])

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

# quantum gates
X = NOT
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0],[0, -1]])
H = np.array([[1, 1], [1, -1]]) * math.sqrt(0.5)



# f_i determines function f. 
# 0: f(x) = 0,           1: f(x) = 1
# 2: f(0) = 0, f(1) = 1, 3: f(0) = 1, f(1) = 0
def deutsch(f_i):
    q0_0 = np.array([0, 1])
    q1_0 = np.array([1, 0])

    q0_1 = np.matmul(H, q0_0)
    q1_1 = np.matmul(H, q1_0)

    match f_i:
        case 0:
            q1_2 = q1_1
        case 1:
            q1_2 = np.matmul(X, q1_1)
        case 2:
            q1_2 = np.matmul(CNOT, np.kron(q0_1, q1_1))[2:]
        case 3: 
            q1_2 = np.matmul(CNOT, np.kron(q0_1, q1_1))[2:]
            q1_2 = np.matmul(X, q1_2)

    q0_3 = np.matmul(H, q0_1)
    q1_3 = q1_2

    print(q0_3)



U_0 = np.eye(4)
U_1 = np.kron(I, NOT)
U_2 = CNOT
U_3 = np.array([[0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

def deutsch2(U):
    p0 = np.kron([1, 0], [0, 1])
    p1 = np.matmul(np.kron(H, H), p0)
    p2 = np.matmul(U, p1)
    p3 = np.matmul(np.kron(H, I), p2)
    print(p3)

print('U1: ')
deutsch2(U_1)
print('U2: ')
deutsch2(U_2)

# print(np.kron(CNOT, I))
# print(np.kron(I, CNOT))

U_4 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]])



# p_i = number of top input qubits, U input function
def deutsch_jozsa(p_i, U):
    p0 = [0, 1]
    g1 = H
    for i in range(0, p_i):
        p0 = np.kron([1, 0], p0)
        g1 = np.kron(H, g1)
    p1 = np.matmul(p0, g1)
    p2 = np.matmul(U, p1)
    p3 = np.matmul(np.kron(H, np.eye(2*p_i)), p2)
    print(p3)

deutsch_jozsa(1, U_2)
deutsch_jozsa(2, U_4)
    
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])
CZ = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, -1]])

def teleport(q):
    bell = np.array([1, 0, 0, 1]) * math.sqrt(2)
    p0 = np.kron(q, bell)
    p1 = np.matmul(np.kron(CNOT, I), p0)
    p2 = np.matmul(np.kron(np.kron(H, I), I), p1)
    p3 = np.matmul(np.kron(I, CNOT), p2)
    p4 = np.matmul(np.kron(I, SWAP), p3)
    p5 = np.matmul(np.kron(CZ, I), p4)
    print(p5)

#TODO: measurement, QFT
def shor(m, n, U):
    p0 = [1, 0]
    g1 = H
    for i in range(0, m+n-1):
        p0 = np.kron([1, 0], p0)
    for i in range(0, m-1):
        g1 = np.kron(H, g1)
    g1 = np.kron(g1, np.eye(2*n))
    p1 = np.matmul(g1, p0)
    p2 = np.matmul(U, p1)