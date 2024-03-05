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
    p3 = np.matmul(np.kron(H, np.eye(2 ** p_i)), p2)
    print(p3)

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


# tq = target qubit, cq = control qubit (both zero-indexed)
# gate = 2x2 gate applied on target qubit, n = total qubits
# output: 2^n x 2^n matrix representing controlled gate operating on tq
def cgate(tq, cq, gate, n):
    if(tq == cq):
        return
    # projectors onto |0> and |1> state
    proj0 = np.array([[1, 0], [0, 0]])
    proj1 = np.array([[0, 0], [0, 1]])
    if tq > cq:
        # unrelated qubits before cq
        mat = np.eye(2 ** cq)
        # qubits between control and target, inclusive
        cont = np.kron(proj0, np.eye(2 ** (tq - cq))) + \
        np.kron(proj1, np.kron(np.eye(2 ** (tq - cq - 1)), gate))
        mat = np.kron(mat, cont)
        # unrelated qubits after tq
        mat = np.kron(mat, np.eye(2 ** (n - tq - 1))) # check off by one case
    else:
        # unrelated qubits before tq
        mat = np.eye(2 ** tq)
        # qubits between target and control, inclusive
        # I reversed the order of the projectors and the target qubit
        cont = np.kron(np.eye(2 ** (cq - tq)), proj0) + \
        np.kron(gate, np.kron(np.eye(2 ** (cq - tq - 1)), proj1))
        mat = np.kron(mat, cont)
        # unrelated qubits after cq
        mat = np.kron(mat, np.eye(2 ** (n - cq - 1)))
    return mat


# tqs to tqf = target qubits, cq = control qubit (both zero-indexed)
# gate = 2x2 gate applied on target qubit, n = total qubits
# output: 2^n x 2^n matrix representing controlled gate operating on tq
def cgate1(tqs, tqf, cq, gate, n):
    # projectors onto |0> and |1> state
    proj0 = np.array([[1, 0], [0, 0]])
    proj1 = np.array([[0, 0], [0, 1]])
    gate_size = tqf - tqs + 1
    if tqs > cq:
        # unrelated qubits before cq
        mat = np.eye(2 ** cq)
        # qubits between control and target, inclusive
        cont = np.kron(proj0, np.eye(2 ** (tqs - cq))) + \
                np.kron(proj1, np.kron(np.eye(2 ** (tqs - cq - gate_size)), gate))
        mat = np.kron(mat, cont)
        # unrelated qubits after tq
        mat = np.kron(mat, np.eye(2 ** (n - tqf - 1))) 
    elif tqf < cq:
        # unrelated qubits before tq
        mat = np.eye(2 ** tqf)
        # qubits between target and control, inclusive
        # I reversed the order of the projectors and the target qubit
        cont = np.kron(np.eye(2 ** (cq - tqs)), proj0) + \
                np.kron(gate, np.kron(np.eye(2 ** (cq - tqs - gate_size)), proj1))
        mat = np.kron(mat, cont)
        # unrelated qubits after cq
        mat = np.kron(mat, np.eye(2 ** (n - cq - 1)))
    return mat

# print(cgate1(0, 0, 2, NOT, 3))

def swap_helper(a, b, n, i, cur_mat):
    if(i == 0):
        return cur_mat

    m = np.zeros(2 ** n)
    return m


def swap(a, b, n):
    swap(a, b, n-1)
    return


# a, b = qubits to be swapped (zero-indexed)
# n = total number of qubits
# output: 2^n x 2^n matrix representing swap gate
def swap1(a, b, n):
    gate = cgate(a, b, NOT, n)
    gate = np.matmul(cgate(b, a, NOT, n), gate)
    gate = np.matmul(cgate(a, b, NOT, n), gate)
    return gate

# print(swap1(1, 2, 3))


# I think I'm applying the phase gates in the opposite order
# that I should, so I'm commenting this out.
'''
def iqft(n):
    m = np.eye(2 ** n)
    for i in range(0, math.ceil((n-1)/2)):
        m = np.matmul(swap1(i, n-1-i, n), m)

    for i in range(0, n-1):
        # H gate on (n-i-1)th qubit
        m = np.matmul(np.kron(np.eye(2 ** (n-i-1)), np.kron(H, np.eye(2 ** i))), m)
        for j in range(0, i+1):
            phase = (-1) * math.pi / (2 ** (i - j + 1))
            phase_gate = np.array([[1, 0],
                                  [0, np.exp(1j*phase)]])
            op = cgate(n-i-2,n-j-1, phase_gate, n)
            m = np.matmul(op, m)
    return m
'''

# n = total qubits
# output: 2^n x 2^n matrix representing the QFT
def qft(n):
    m = np.eye(2 ** n)
    for i in range(0, n-1):
        # H gate on ith qubit
        m = np.matmul(np.kron(np.eye(2 ** i), np.kron(H, np.eye(2 ** (n-i-1)))), m)
        for j in range(0, i+1):
            # R_j gate
            phase = math.pi / (2 ** (i - j + 1))
            phase_gate = np.array([[1, 0],
                                  [0, np.exp(1j*phase)]])
            
            op = cgate(j, i+1, phase_gate, n)
            m = np.matmul(op, m)
    # H gate on last qubit
    m = np.matmul(np.kron(np.eye(2 ** (n-1)), H), m)
    # swap qubit order
    for i in range(0, math.ceil((n-1)/2)):
        m = np.matmul(swap1(i, n-1-i, n), m)
    return m

# print(np.ndarray.round(qft(3), 3))


# t = number of measurement qubits, u = number of gate qubits
# u_vec = initialized eigenvector, U = target gate
# output: a 2^(t+u) vector representing the qubit vector state
def phase_est(t, u, u_vec, U):
    # TODO: implement qubit measurement
    # ideally, I would like to output the phase itself, not the qubit vector
    # initialize measurement qubits to |+>
    t_init = np.array([math.sqrt(0.5), math.sqrt(0.5)])
    for i in range(1, t):
        t_init = np.kron(np.array([math.sqrt(0.5), math.sqrt(0.5)]), t_init)
    # combine measurement + gate qubits into one qubit vector
    p = np.kron(t_init, u_vec)
    # apply controlled-U gate in series
    for i in range(0, t):
        for j in range(0, 2 ** i):
            p = np.matmul(cgate1(t, t+u-1, t-i-1, U, t+u), p)
    
    # inverse quantum Fourier transform
    qft_mat = qft(t)
    iqft_mat = np.ndarray.conjugate(np.ndarray.transpose(qft_mat))
    p = np.matmul(np.kron(iqft_mat, np.eye(2 ** u)), p)
    return p

ex_gate = np.array([[0, -1j],
                    [1j, 0]])
eigenvec = np.array([1, -1j]) * math.sqrt(0.5)

print(np.ndarray.round(phase_est(2, 1, eigenvec, ex_gate), 3))

# the correct output is |1>|0>|eigenvec>
# print(np.ndarray.round(np.kron(np.kron(np.array([0, 1]), np.array([1, 0])), eigenvec), 3))

#print(np.ndarray.round(iqft(2), 3))
#print(np.ndarray.round(np.matmul(qft(2), iqft(2)), 3))

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