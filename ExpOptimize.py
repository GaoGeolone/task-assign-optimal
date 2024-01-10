import numpy as np
import random
import sys
from pyLingo import * # this package use numpy as N 
import csv
from mako.template import Template

def floyd_warshall(adj_matrix):
    n = len(adj_matrix)
    
    # 初始化距离矩阵为邻接矩阵的副本，同时将不存在的路径设为inf
    distances = [row[:] for row in adj_matrix]
    for i in range(n):
        for j in range(n):
            if distances[i][j] == 0 and i != j:
                distances[i][j] = float('inf')

    # 初始化路径矩阵为节点i到节点j的直接路径
    paths = [[[] if adj_matrix[i][j] == 0 else [i, j] for j in range(n)] for i in range(n)]
    
    # 利用Floyd-Warshall算法更新最短路径和距离
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]
                    paths[i][j] = paths[i][k] + paths[k][j][1:]
    
    return distances, paths


def replace_element(adj_matrix, target_value, replace_value):
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] == target_value:
                adj_matrix[i][j] = replace_value


def DefinedKroneckerProduct(A,N):
    C00 = A[0:3,0:3]
    C01 = A[0:3,3:6]
    C10 = A[3:6,0:3]
    C11 = A[3:6,3:6]
    # np.ones((3,3)) np.eyes(3)
    return np.block([[np.kron(C00,np.eye(N[0])),np.kron(C01,np.ones((N[0],N[1])))],
                [np.kron(C10,np.ones((N[1],N[0]))),np.kron(C11,np.eye(N[1]))]])

def CBGProblem(TestCounter,Ncu,Nlk,Nf,Nprop,tensor,fluxfpair,workloadf,bandwidth,speed,delay,delayconstraint,objconstraint,InitXP,scriptfile = "CBGProblem.lng"):

    # 定义变量
    # cu = np.array([list(range(Ncu))])
    # lk = np.array([list(range(Nlk))])
    # f = np.array([list(range(Nf))])
    # p = np.array([list(range(Nprop))])
    #create Lingo enviroment object
    pEnv = lingo.pyLScreateEnvLng()
    if pEnv is None:
        print("cannot create LINGO environment!")
        exit(1)

    #open LINGO's log file
    errorcode = lingo.pyLSopenLogFileLng(pEnv,f'CBG-{TestCounter}.log')
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #pass memory transfer pointers to LINGO
    #define pnPointersNow
    pnPointersNow = np.array([0],dtype=np.int32)

    #@POINTER(1)
    tensor_arr = np.reshape(tensor,(-1))
    tensor_1 = np.array(tensor_arr,dtype=np.double)

    errorcode = lingo.pyLSsetDouPointerLng(pEnv, tensor_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(2)
    fluxfpair_arr = np.reshape(fluxfpair,(-1))
    fluxfpair_1 = np.array(fluxfpair_arr,dtype=np.double)

    errorcode = lingo.pyLSsetDouPointerLng(pEnv, fluxfpair_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(3)
    workloadf_arr = np.reshape(workloadf,(-1))
    workloadf_1 = np.array(workloadf_arr,dtype=np.double)

    errorcode = lingo.pyLSsetDouPointerLng(pEnv, workloadf_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(4)
    print(bandwidth)
    bandwidth_arr = np.reshape(bandwidth,(-1))
    bandwidth_1 = np.array(bandwidth_arr,dtype=np.double)
    
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, bandwidth_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(5)
    speed_arr = np.reshape(speed,(-1))
    speed_1 = np.array(speed_arr,dtype=np.double)

    errorcode = lingo.pyLSsetDouPointerLng(pEnv, speed_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(6)
    delay_arr = np.reshape(delay,(-1))
    delay_1 = np.array(delay_arr,dtype=np.double)

    errorcode = lingo.pyLSsetDouPointerLng(pEnv, delay_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(7)
    delayconstraint_arr = np.reshape(delayconstraint,(-1))
    delayconstraint_1 = np.array(delayconstraint_arr,dtype=np.double)

    errorcode = lingo.pyLSsetDouPointerLng(pEnv, delayconstraint_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(8)
    objconstraint_arr = np.reshape(objconstraint,(-1))
    objconstraint_1 = np.array(objconstraint_arr,dtype=np.double)

    errorcode = lingo.pyLSsetDouPointerLng(pEnv, objconstraint_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)    

    #@POINTER(9)
    InitXP_arr = np.reshape(InitXP,(-1))
    Xp = np.array(InitXP_arr,dtype=np.double)
    print(np.shape(Xp))
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, Xp, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(10)
    MaxRatio = np.array([-1.0],dtype=np.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, MaxRatio, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(11)
    Status = np.array([-1.0],dtype=np.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, Status, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #Run the script

    cScript = f"SET ECHOIN 1 \n TAKE {scriptfile} \n GO \n QUIT \n"
    errorcode = lingo.pyLSexecuteScriptLng(pEnv, cScript)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #Close the log file
    errorcode = lingo.pyLScloseLogFileLng(pEnv)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    if Status[0] == const.LS_STATUS_GLOBAL_LNG:
        print("\nGlobal optimum found!")
    elif Status[0] == const.LS_STATUS_LOCAL_LNG:
        print("\nLocal optimum found!")
    else:
        print("\nSolution is non-optimal\n")

    #check solution
    Xp = Xp.reshape(Ncu,-1)
    print("\nThe Optimal solution \n",Xp.reshape(Ncu,-1),".\n Result", 
           MaxRatio," .\n\n")

    #delete Lingo enviroment object
    errorcode = lingo.pyLSdeleteEnvLng(pEnv)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        exit(1)

    print(CalcloadOfCUandLK(TestCounter,Ncu,Nlk,Nf,Nprop,tensor,fluxfpair,workloadf,bandwidth,speed,delay,delayconstraint,Xp))

def CalcloadOfCUandLK(TestCounter,Ncu,Nlk,Nf,Nprop,tensor,fluxfpair,workloadf,bandwidth,speed,delay,delayconstraint,XP):
    tr_flux = np.dot(np.dot(XP,fluxfpair),XP.T)
    lk_load = np.zeros(Nlk)
    for i in range(0,Nlk):
        lk_load[i] = np.tensordot(tensor[i, :, :], tr_flux, axes=2)/bandwidth[i]
    cu_load = np.dot(np.dot(XP,workloadf),np.ones(Nf))
    for j in range(0,Ncu):
        cu_load[j] = cu_load[j]/speed[j]

    print('lk_load:',lk_load)
    print('cu_load:',cu_load)
    return lk_load

def InitXp(Ncu, Nf):
    Xp = np.zeros((Ncu,Nf))
    cui = 0
    for idf in range(Nf):
        if idf % 3 == 0:
            cui = random.randint(0,Ncu-1)
        Xp[cui][idf] = 1
    return Xp

def InitXe(Ncu, Ne):
    Xe = np.zeros((Ncu,Ne))
    for ide in range(Ne):
        Xe[random.randint(0,Ncu-1)][ide] = 1
    return Xe

# 0->float('inf')
Adj_matrix = [
    #CU                      LK                                  R      
    #0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0 cu
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #5
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #6
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #7
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #8 lk
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #9
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #10
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #11
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #12
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #13
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #14
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #15
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], #18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], #19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1], #20 r
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], #21
]

replace_element(Adj_matrix, 0, float('inf'))
distances, paths = floyd_warshall(Adj_matrix)

print("最短路径距离矩阵:")
for row in distances:
    print(row)

# 创建一个三阶张量（14 x 8 x 8）
shape = (14, 8, 8)
tensor = np.zeros(shape)

# #Only for test
# # 假设有两个矩阵A和B
# A = np.array([[1, 2], [3, 4]])
# B = np.array([[5, 6], [7, 8]])

# 计算Frobenius张量内积
# inner_product = np.tensordot(A, B, axes=2)

# 打印结果
# print(inner_product)
# End of test

print("\n每对节点之间的最短路径:")
for i in range(len(paths)):
    for j in range(len(paths[i])):
        if i in range(0,8) and j in range(0,8):
            print(f"从节点{i}到节点{j}: {paths[i][j]}")
            for tn in paths[i][j]:
                tensor[tn-8][i][j] = 1
                tensor[tn-8][j][i] = 1

# print(tensor)

# Publish Matrix of Class
EPC = np.array([
    [1, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0]
])
# Subscribe Matrix of Class
ESC = np.array([
    [1, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0]
])
# Entity Number C0 with 10 C1 with 20
N = np.array([10,10]) #Note: Affect F and P Number in LNG file
# Subscribe Matrix of Entities
EP = DefinedKroneckerProduct(EPC,N)
# Subscribe Matrix of Entities(Special)
C00a = ESC[0:3,0:1]
C00b = ESC[0:3,1:3]
C01 = ESC[0:3,3:6]
C10 = ESC[3:6,0:3]
C11a = ESC[3:6,3:4]
C11b = ESC[3:6,4:6]
ES = np.block([[np.kron(C00a,np.eye(N[0])),np.kron(C00b,np.ones((N[0],N[0]))),np.kron(C01,np.ones((N[0],N[1])))],
                [np.kron(C10,np.ones((N[1],N[0]))),np.kron(C11a,np.eye(N[1])),np.kron(C11b,np.ones((N[1],N[1])))]])
# Data Size of Entities's state
DxC = np.diag([10,35,20,10,40,10])
Dx = DefinedKroneckerProduct(DxC,N)
# Work Load of Entities's functions
WfC = np.diag([100,300,100,30,140,100])
Wf = DefinedKroneckerProduct(WfC,N)
# print(Wf)
#Speed of each CU
SP = np.array([100,100,100,100,100,100,100,100])
#Bandwitdh of each LK
BW = np.array([1000,1000,1000,1000,1000,1000,1000,1000,100,100,100,100,10,10])
Delay = np.array([0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0015,0.0015,0.0015,0.0015,0.015,0.015])
# Define function frequencies
BasicFreq = np.array([10,1,15,20,2,15]) # 5dimensions
Omega = DefinedKroneckerProduct(np.diag(BasicFreq),N)
# Define delay constraint for functions
SizeF = np.shape(Omega)
# Use default delay constraint 1/freq
DelayF = np.zeros(SizeF)
for i in range(SizeF[0]):
    for j in range(SizeF[1]):
        DelayF[i][j] = 1/Omega[j][j]
# Define object co-assign constraint
DefaultCoAssign = np.zeros(SizeF)
FedCoAssign = np.kron(np.eye(N[0]+N[1]),np.ones((3,3)))
# print(FedCoAssign)

Ncu = shape[1]
Nlk = shape[0]
Nf = SizeF[0]
Nprop = np.shape(Dx)[0]
# 计算矩阵乘法
fluxfpair = np.dot(np.dot(np.dot(Omega,EP),Dx),ES)
Xp = InitXp(Ncu,Nf)
Xe = InitXe(Ncu,np.sum(N))
# 定义模板文件路径和生成的 Lingo 脚本文件路径
template_path = 'CBGProblem.mako'
output_script_path = 'CBGProblem.lng'
# 从模板文件渲染模板并生成 Lingo 脚本
template = Template(filename=template_path)
output_script = template.render(
    NCU=f'{Ncu}',
    NLK=f'{Nlk}',
    Nf=f'{Nf}',
    Np=f'{Nprop}'
)
# 将生成的脚本写入文件
with open(output_script_path, 'w') as f:
    f.write(output_script)

columns = ['count','moegaf0','moegag0','moegaf1','moegag1','moegal0','obj','Xp']
data = [1,10,1,20,2,15]
CalcloadOfCUandLK(1,Ncu,Nlk,Nf,Nprop,tensor,fluxfpair,Wf,BW,SP,Delay,DelayF,Xp)
CBGProblem(1,Ncu,Nlk,Nf,Nprop,tensor,fluxfpair,Wf,BW,SP,Delay,DelayF,FedCoAssign,Xp,scriptfile=output_script_path)
