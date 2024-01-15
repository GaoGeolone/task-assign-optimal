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

def GenerateTensorFromAdj(Adj_matrix,shape):
    replace_element(Adj_matrix, 0, float('inf'))
    distances, paths = floyd_warshall(Adj_matrix)

    print("最短路径距离矩阵:")
    for row in distances:
        print(row)

    # 创建一个三阶张量（14 x 8 x 8）
    tensor = np.zeros(shape)

    print("\n每对节点之间的最短路径:")
    for i in range(len(paths)):
        for j in range(len(paths[i])):
            if i in range(shape[1]) and j in range(shape[2]):
                print(f"从节点{i}到节点{j}: {paths[i][j]}")
                for tn in paths[i][j]:
                    if tn >= shape[1]:
                        tensor[tn-shape[1]][i][j] = 1
                        tensor[tn-shape[1]][j][i] = 1
    return tensor

def DefinedKroneckerProduct(A,N):
    C00 = A[0:3,0:3]
    C01 = A[0:3,3:6]
    C10 = A[3:6,0:3]
    C11 = A[3:6,3:6]
    # np.ones((3,3)) np.eyes(3)
    return np.block([[np.kron(C00,np.eye(N[0])),np.kron(C01,np.ones((N[0],N[1])))],
                [np.kron(C10,np.ones((N[1],N[0]))),np.kron(C11,np.eye(N[1]))]])

def CBGProblem(TestCounter,Ncu,Nlk,Nf,Nprop,tensor,fluxfpair,Frequency,workloadf,bandwidth,speed,delay,delayconstraint,objconstraint,InitXP,scriptfile = "CBGProblem.lng"):

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
    errorcode = lingo.pyLSopenLogFileLng(pEnv,f'./Log/CBG-{TestCounter}.log')
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
    Frequency_arr = np.reshape(Frequency,(-1))
    Frequency_1 = np.array(Frequency_arr,dtype=np.double)
    print(Frequency)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, Frequency_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(4)
    workloadf_arr = np.reshape(workloadf,(-1))
    workloadf_1 = np.array(workloadf_arr,dtype=np.double)
    # print(np.diag(workloadf))
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, workloadf_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(5)
    bandwidth_arr = np.reshape(bandwidth,(-1))
    bandwidth_1 = np.array(bandwidth_arr,dtype=np.double)
    
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, bandwidth_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(6)
    speed_arr = np.reshape(speed,(-1))
    speed_1 = np.array(speed_arr,dtype=np.double)

    errorcode = lingo.pyLSsetDouPointerLng(pEnv, speed_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(7)
    delay_arr = np.reshape(delay,(-1))
    delay_1 = np.array(delay_arr,dtype=np.double)

    errorcode = lingo.pyLSsetDouPointerLng(pEnv, delay_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(8)
    delayconstraint_arr = np.reshape(delayconstraint,(-1))
    delayconstraint_1 = np.array(delayconstraint_arr,dtype=np.double)

    errorcode = lingo.pyLSsetDouPointerLng(pEnv, delayconstraint_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(9)
    objconstraint_arr = np.reshape(objconstraint,(-1))
    objconstraint_1 = np.array(objconstraint_arr,dtype=np.double)

    errorcode = lingo.pyLSsetDouPointerLng(pEnv, objconstraint_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)    

    #@POINTER(10)
    InitXP_arr = np.reshape(InitXP,(-1))
    Xp = np.array(InitXP_arr,dtype=np.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, Xp, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(11)
    MaxRatio = np.array([-1.0],dtype=np.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, MaxRatio, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(12)
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

    SolutionQuality = 'None'
    if Status[0] == const.LS_STATUS_GLOBAL_LNG:
        print("\nGlobal optimum found!")
        SolutionQuality = 'Global'
    elif Status[0] == const.LS_STATUS_LOCAL_LNG:
        print("\nLocal optimum found!")
        SolutionQuality = 'Local'
    else:
        print("\nSolution is non-optimal\n")
        SolutionQuality = 'non-optimal'

    #check solution
    Xp = Xp.reshape(Ncu,-1)
    print("\nThe Optimal solution \n",Xp,".\n Optimal Result", 
           MaxRatio," (Not Real Workload for cu and lk).\n\n")

    #delete Lingo enviroment object
    errorcode = lingo.pyLSdeleteEnvLng(pEnv)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        exit(1)
    fluxfpairOmega = np.dot(np.diag(Frequency),fluxfpair)
    Obj = CalcloadOfCUandLK(TestCounter,Ncu,Nlk,Nf,Nprop,tensor,fluxfpairOmega,workloadf,bandwidth,speed,delay,delayconstraint,Xp)
    # IF Obj > Ratio ,must be the case Only Cu or OnlyLk
    print(f'Real load Result: [{Obj}]')
    return [Obj, Xp, SolutionQuality]

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

    return np.amax(np.concatenate((lk_load, cu_load), axis=0))

def InitXp(Ncu, Nf):
    Xp = np.zeros((Ncu,Nf))
    cui = 0
    for idf in range(Nf):
        if idf % 3 == 0:
            cui = random.randint(Ncu,Ncu-1)
        Xp[cui][idf] = 1
    return Xp

def InitXe(Ncu, Ne):
    Xe = np.zeros((Ncu,Ne))
    for ide in range(Ne):
        Xe[random.randint(0,Ncu)][ide] = 1
    return Xe


def IteratorOfOptimal(count, shape, tensor, parametre, Xp, FedCoAssign):
    N0,N1,omegaf0,omegag0,omegaf1,omegag1,omegal0,Type,ld0,ld1,ld2,ld3,ld4 = parametre
    # print(f'{omegaf0},{omegag0},{omegaf1},{omegag1},{omegal0},{Type}')
    # Entity Number C0 with 10 C1 with 20
    N = np.array([N0,N1]) #Note: Affect F and P Number in LNG file
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
    DxC = np.diag([5,30,10,15,15,10])
    Dx = DefinedKroneckerProduct(DxC,N)
    # Define function frequencies N0,N1,omegaf0,omegag0,omegaf1,omegag1,omegal0,Type = 10,1,15,20,2,15parametre
    BasicFreq = np.array([omegaf0,omegag0,omegal0,omegaf1,omegag1,omegal0]) # 5dimensions
    Omega = DefinedKroneckerProduct(np.diag(BasicFreq),N)
    FreqForF = np.diag(Omega)  #diag

    # Work Load of Entities's functions
    WfC = np.diag([ld0,ld1,ld4,ld2,ld3,ld4])
    Wf = DefinedKroneckerProduct(WfC,N)
    Wf = np.dot(Wf,Omega)
    # print(Wf)
    #Speed of each CU
    SP = np.array([1000,1000,1000,1000,1000,1000,1000,1000])
    #Bandwitdh of each LK
    BW = np.array([2000,2000,2000,2000,2000,2000,2000,2000,500,500,500,500,125,125])
    Delay = np.array([1.0e-5,1.0e-5,1.0e-5,1.0e-5,1.0e-5,1.0e-5,1.0e-5,1.0e-5,1.0e-4,1.0e-4,1.0e-4,1.0e-4,1.0e-3,1.0e-3])

    # Define delay constraint for functions
    SizeF = np.shape(Omega)
    # Use default delay constraint 1/freq
    DelayF = np.zeros(SizeF)
    for i in range(SizeF[0]):
        for j in range(SizeF[1]):
            DelayF[i][j] = 1/Omega[j][j]
    # print(FedCoAssign)


    Ncu = shape[1]
    Nlk = shape[0]
    Nf = SizeF[0]
    Nprop = np.shape(Dx)[0]
    # 计算矩阵乘法 Not Calculate Omega here but in lingo
    fluxfpair = np.dot(np.dot(EP,Dx),ES)
    # But random mode need fluxfpair with Freq
    fluxfpairOmega = np.dot(Omega,fluxfpair)# Row product
    
    # 定义模板文件路径和生成的 Lingo 脚本文件路径
    template_path = 'CCBGProblem.mako'
    output_script_path = 'CCBGProblem.lng'
    # 从模板文件渲染模板并生成 Lingo 脚本
    if Type == 'FunAssigned':
        # Function Assigned
        Xedef = '! Not adopted;'
        EntityTaskAssignmentMode = '! Not adopted;'
        XConstraint = '@FOR(fv(k):@SUM(cuv(ci):Xp(ci,k))=1); ! Function Task assignment matrix;\n@FOR(cuxf(i,j):@BIN(Xp(i,j))); ! Function Task assignment matrix;'
        WorkloadOfEachTransportNode = '@FOR(lv(k):MaxRatio >= @SUM(cuxcu(i,j):\nTENSOR(k,i,j)*@SUM(fxf(fi,fj):\nXp(i,fi)*FREQUENCY(fi)*FLUXFORFUNCPAIR(fi,fj)*Xp(j,fj)))/BANDWIDTH(k))  ; '
        WorkloadOfEachComputeUnit = '@FOR(cuv(cui):MaxRatio >=@SUM(fv(lfj):Xp(cui,lfj)*WORKLOAD(lfj,lfj))/SPEED(cui));'
    elif Type == 'EntityAssigned':
        # Entities Assigned
        Xedef = 'cuxe(cu, e): Xe; ! Entity Task assignment matrix;'
        EntityTaskAssignmentMode = '@FOR(cuxf(ci,fj):Xp(ci,fj)=@SUM(ev(ek):Xe(ci,ek)*OBJECTCONSTRAINT(ek,fj)));'
        XConstraint = '@FOR(ev(k):@SUM(cuv(ci):Xe(ci,k))=1); ! Entity Task assignment matrix;\n@FOR(cuxe(i,j):@BIN(Xe(i,j))); ! Entity Task assignment matrix;'
        WorkloadOfEachTransportNode = '@FOR(lv(k):MaxRatio >= @SUM(cuxcu(i,j):TENSOR(k,i,j)*@SUM(fxf(fi,fj):Xp(i,fi)*FREQUENCY(fi)*FLUXFORFUNCPAIR(fi,fj)*Xp(j,fj)))/BANDWIDTH(k))  ; '
        WorkloadOfEachComputeUnit = '@FOR(cuv(cui):MaxRatio >=@SUM(fv(lfj):Xp(cui,lfj)*WORKLOAD(lfj,lfj))/SPEED(cui));'
    elif Type == 'Random':
        Obj = CalcloadOfCUandLK(1,Ncu,Nlk,Nf,Nprop,tensor,fluxfpairOmega,Wf,BW,SP,Delay,DelayF,Xp)
        print(f'Real load Result: [{Obj}]')
        return [Obj,Xp,'non-optimal']
    elif Type == 'FOnlyCU':
        Xedef = '! Not adopted;'
        EntityTaskAssignmentMode = '! Not adopted;'
        XConstraint = '@FOR(fv(k):@SUM(cuv(ci):Xp(ci,k))=1); ! Function Task assignment matrix;\n@FOR(cuxf(i,j):@BIN(Xp(i,j))); ! Function Task assignment matrix;'
        WorkloadOfEachTransportNode = '! Not adopted;'
        WorkloadOfEachComputeUnit = '@FOR(cuv(cui):MaxRatio >=@SUM(fv(lfj):Xp(cui,lfj)*WORKLOAD(lfj,lfj))/SPEED(cui));'
    elif Type == 'FOnlyLK':
        Xedef = '! Not adopted;'
        EntityTaskAssignmentMode = '! Not adopted;'
        XConstraint = '@FOR(fv(k):@SUM(cuv(ci):Xp(ci,k))=1); ! Function Task assignment matrix;\n@FOR(cuxf(i,j):@BIN(Xp(i,j))); ! Function Task assignment matrix;'
        WorkloadOfEachTransportNode = '@FOR(lv(k):MaxRatio >= @SUM(cuxcu(i,j):TENSOR(k,i,j)*@SUM(fxf(fi,fj):Xp(i,fi)*FREQUENCY(fi)*FLUXFORFUNCPAIR(fi,fj)*Xp(j,fj)))/BANDWIDTH(k))  ; '
        WorkloadOfEachComputeUnit = '! Not adopted;'
    elif Type == 'EOnlyCU':
        Xedef = 'cuxe(cu, e): Xe; ! Entity Task assignment matrix;'
        EntityTaskAssignmentMode = '@FOR(cuxf(ci,fj):Xp(ci,fj)=@SUM(ev(ek):Xe(ci,ek)*OBJECTCONSTRAINT(ek,fj)));'
        XConstraint = '@FOR(ev(k):@SUM(cuv(ci):Xe(ci,k))=1); ! Entity Task assignment matrix;\n@FOR(cuxe(i,j):@BIN(Xe(i,j))); ! Entity Task assignment matrix;'
        WorkloadOfEachTransportNode = '! Not adopted;'
        WorkloadOfEachComputeUnit = '@FOR(cuv(cui):MaxRatio >=@SUM(fv(lfj):Xp(cui,lfj)*WORKLOAD(lfj,lfj))/SPEED(cui));'
    elif Type == 'EOnlyLK':
        Xedef = 'cuxe(cu, e): Xe; ! Entity Task assignment matrix;'
        EntityTaskAssignmentMode = '@FOR(cuxf(ci,fj):Xp(ci,fj)=@SUM(ev(ek):Xe(ci,ek)*OBJECTCONSTRAINT(ek,fj)));'
        XConstraint = '@FOR(ev(k):@SUM(cuv(ci):Xe(ci,k))=1); ! Entity Task assignment matrix;\n@FOR(cuxe(i,j):@BIN(Xe(i,j))); ! Entity Task assignment matrix;'
        WorkloadOfEachTransportNode = '@FOR(lv(k):MaxRatio >= @SUM(cuxcu(i,j):TENSOR(k,i,j)*@SUM(fxf(fi,fj):Xp(i,fi)*FREQUENCY(fi)*FLUXFORFUNCPAIR(fi,fj)*Xp(j,fj)))/BANDWIDTH(k))  ; '
        WorkloadOfEachComputeUnit = '! Not adopted;'

    template = Template(filename=template_path)
    output_script = template.render(
        NCU=f'{Ncu}',
        NLK=f'{Nlk}',
        Nf=f'{Nf}',
        Np=f'{Nprop}',
        Ne = f'{np.sum(N)}',
        Xedef = Xedef,
        EntityTaskAssignmentMode=EntityTaskAssignmentMode,
        XConstraint = XConstraint,
        WorkloadOfEachComputeUnit = WorkloadOfEachComputeUnit,
        WorkloadOfEachTransportNode = WorkloadOfEachTransportNode
    )
    # 将生成的脚本写入文件
    with open(output_script_path, 'w') as f:
        f.write(output_script)


    # CalcloadOfCUandLK(1,Ncu,Nlk,Nf,Nprop,tensor,fluxfpair,Wf,BW,SP,Delay,DelayF,Xp)
    return CBGProblem(count,Ncu,Nlk,Nf,Nprop,tensor,fluxfpair,FreqForF,Wf,BW,SP,Delay,DelayF,FedCoAssign,Xp,scriptfile=output_script_path)


# 0->float('inf')
Adj_matrix = [
    #CU                      LK0 1 2  3  4  5  6  7  8  9  10 11 R0 R1      
    #0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #0 cu
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #1
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], #4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], #5
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], #6
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #7
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #8 lk0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #9 lk1
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #10 lk2
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #11 lk3
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #12 lk4
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #13 lk5
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #14 lk6
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], #15 lk7
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #16 lk8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], #17 lk9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], #18 lk10
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], #19 lk11
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1], #20 r
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], #21
]


shape = (14, 8, 8)  
tensor = GenerateTensorFromAdj(Adj_matrix,shape)
count = 0
columns = ['count','omegaf0','omegag0','omegaf1','omegag1','omegal0','Type','obj','SolutionQuality','Xp']
rows=[]
random.seed(18)
# 使用'w'模式创建文件对象，定义newline参数可以避免写入空行
with open('dataFreq.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)
        # Xp = InitXp(Ncu,Nf)
    N = [2,2]
    ClassFun = [[1, 1, 1, 0, 0, 0],[0, 0, 0, 1, 1, 1]]
    # print(ClassFun[1])
    FedCoAssign = np.block([[np.kron(ClassFun[0],np.eye(N[0]))],[np.kron(ClassFun[1],np.eye(N[1]))]])

    print(FedCoAssign)
    Xe = InitXe(8,np.sum(N))
    Xp = np.dot(Xe,FedCoAssign)
    # Define object co-assign constraint
    for w0 in [10,15,20,25,30,35,40,45,50]:
        for w1 in [10,15,20,25,30,35,40,45,50]:
            w2 = 35
            w3 = 35
            ld0 = 300
            ld1 = 100
            ld2 = 100
            ld3 =100
            ld4 = 100
            # for w2 in [1,5,10,15,20,25,30,35]:
                # for w3 in [1,5,10,15,20,25,30,35]:
            for w4 in [10,15,20,25,30,35,40,45,50]:
                for Type in ['FunAssigned','EntityAssigned','FOnlyCU','FOnlyLK','EOnlyCU','EOnlyLK','Random']:
                    # set Param
                    parametre=[N[0],N[1],w0,w1,w2,w3,w4,Type,ld0,ld1,ld2,ld3,ld4]
                    print(f'test in count:{count}')
                    [obj,xp,Quality]=IteratorOfOptimal(count, shape, tensor, parametre, Xp, FedCoAssign)
                    row = [count,w0,w1,w2,w3,w4,Type,obj,Quality,xp]
                    rows.append(row)
                    count = count + 1

    for row in rows:
        writer.writerow(row)
rows = []
columns = ['count','workloadf0','workloadg0','workloadf1','workloadg1','oworkloadl0','Type','obj','SolutionQuality','Xp']
with open('datawf.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(columns)
        # Xp = InitXp(Ncu,Nf)
    N = [2,2]
    ClassFun = [[1, 1, 1, 0, 0, 0],[0, 0, 0, 1, 1, 1]]
    # print(ClassFun[1])
    FedCoAssign = np.block([[np.kron(ClassFun[0],np.eye(N[0]))],[np.kron(ClassFun[1],np.eye(N[1]))]])

    print(FedCoAssign)
    Xe = InitXe(8,np.sum(N))
    Xp = np.dot(Xe,FedCoAssign)
    # Define object co-assign constraint
    for ld0 in [100,150,200,250,300,350,400,450,500,550]:
        for ld1 in [100,150,200,250,300,350,400,450,500,550]:
            w0 = 35
            w1 = 35
            w2 = 35
            w3 = 35
            w4 = 35
            ld2 = 200
            ld3 =200
            # for w2 in [1,5,10,15,20,25,30,35]:
                # for w3 in [1,5,10,15,20,25,30,35]:
            for ld4 in [100,150,200,250,300,350,400,450,500,550]:
                for Type in ['FunAssigned','EntityAssigned','FOnlyCU','FOnlyLK','EOnlyCU','EOnlyLK','Random']:
                    # set Param
                    parametre=[N[0],N[1],w0,w1,w2,w3,w4,Type,ld0,ld1,ld2,ld3,ld4]
                    print(f'test in count:{count}')
                    [obj,xp,Quality]=IteratorOfOptimal(count, shape, tensor, parametre, Xp, FedCoAssign)
                    row = [count,ld0,ld1,ld2,ld3,ld4,Type,obj,Quality,xp]
                    rows.append(row)
                    count = count + 1

    for row in rows:
        writer.writerow(row)