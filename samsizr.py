#  A Python programming example of interfacing with LINGO API.
#  An application to the Acceptance Sampling Design.
import sys
from pyLingo import *

def samsizr(AQL,LTFD,PRDRISK,CONRISK,MINSMP,MAXSMP):

    # 定义变量
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

    #create Lingo enviroment object
    pEnv = lingo.pyLScreateEnvLng()
    if pEnv is None:
        print("cannot create LINGO environment!")
        exit(1)

    #open LINGO's log file
    errorcode = lingo.pyLSopenLogFileLng(pEnv,'samsizr.log')
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #pass memory transfer pointers to LINGO
    #define pnPointersNow
    pnPointersNow = N.array([0],dtype=N.int32)
    
    #@POINTER(1)
    AQL_1 = N.array([AQL],dtype=N.double)
    print(AQL_1)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, AQL_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(2)
    LTFD_1 = N.array([LTFD],dtype=N.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, LTFD_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(3)
    PRDRISK_1 = N.array([PRDRISK],dtype=N.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, PRDRISK_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(4)
    CONRISK_1 = N.array([CONRISK],dtype=N.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, CONRISK_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(5)
    MINSMP_1 = N.array([MINSMP],dtype=N.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, MINSMP_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(6)
    MAXSMP_1 = N.array([MAXSMP],dtype=N.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, MAXSMP_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(7)
    a = N.array([[1.0,2.0,3.0,4.0],
                [5.0,6.0,7.0,8.0]])
    b = N.reshape(a,(-1))
    print(b)
    a_1 = N.array(b,dtype=N.double)
    print(a_1)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, a_1, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(8)
    NN = N.array([-1.0],dtype=N.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, NN, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(9)
    C = N.array([-1.0],dtype=N.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, C, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #@POINTER(10)
    Status = N.array([-1.0],dtype=N.double)
    errorcode = lingo.pyLSsetDouPointerLng(pEnv, Status, pnPointersNow)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        print("errorcode = ", errorcode)
        exit(1)

    #Run the script
    cScript = "SET ECHOIN 1 \n TAKE samsizr.lng \n GO \n QUIT \n"
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
    print("\nThe Optimal sample size is ",NN,".\nAccept the lot if ",
           C," or less defectives in sample.\n\n")

    #delete Lingo enviroment object
    errorcode = lingo.pyLSdeleteEnvLng(pEnv)
    if errorcode != const.LSERR_NO_ERROR_LNG:
        exit(1)

##############################################################################
if __name__ == '__main__':
    samsizr(0.03,0.08,0.09,0.05,125.0,400.0)
    sys.stdin.read(1)
else:
    print('hahahahahahahaha')

        
