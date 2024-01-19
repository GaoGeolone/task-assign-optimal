MODEL:
! 将Cascade Bipartite Graph的任务分配问题用一个minmax的目标进行优化;
! 使得任务分配对计算和通信节点的时间开销最大值最小化;
! 变量主要为频率omega和任务分配矩阵Xp;
!  Cascade Bipartite Graph;

SETS:
  cu /1..${NCU}/;
  lk /1..${NLK}/;
  f /1..${Nf}/;
  p /1..${Np}/;
  e /1..${Ne}/;
! Make all the vectors, column vectors;
  lxcxc( lk, cu, cu):  TENSOR;
  fxf(f, f): OMEGA,WORKLOAD,DELAYCONSTRAINT,FLUXFORFUNCPAIR;
  lv(lk): BANDWIDTH, DELAY,lkload;
  cuv(cu): SPEED,cuload;
  fv(f): FREQUENCY;
  ev(e): evDemo;
  cuxcu(cu, cu): DELAYFORCU;
! objective function;
  cuxf(cu, f): Xp; ! Function Task assignment matrix;
  ${Xedef}
  exf(e, f): OBJECTCONSTRAINT;
ENDSETS

DATA:
  TENSOR  = @POINTER( 1);     ! "Topology" for different Compute Units with Transport nodes;
  FLUXFORFUNCPAIR = @POINTER( 2);    ! "Data Load" of states;
  FREQUENCY = @POINTER( 3);        ! "Freqency" of f diag([freqency,freqency,...,freqency]);
  WORKLOAD = @POINTER( 4);        ! "Workload" of f diag([workload,workload,...,workload]);
  BANDWIDTH = @POINTER( 5);       ! "Bandwidth" of Transport nodes;
  SPEED = @POINTER( 6);           ! "Speed" of Compute Units;
  DELAY = @POINTER( 7);           ! "Delay" of Transport nodes;
  DELAYCONSTRAINT = @POINTER( 8); ! "Delay" constraint for pair(fi,fj);
  OBJECTCONSTRAINT = @POINTER( 9); ! Bool flag for "Object" constraint for pair(fi,fj);
ENDDATA


procedure TensorProduct1:
! Frobenius inner product x1;
  @FOR(cuxcu(i,j):DELAYFORCU(i,j)=
       @SUM(lv(k):
            TENSOR(k,i,j)*DELAY(k)))  ;
endprocedure

CALC:
  ! Default options;
  @SET( 'DEFAULT');
  ! Suppress default output;   
  ! @SET( 'TERSEO', 2);
  @SET('MTMODE',1);!Set the mode of multithreading;
  @SET('NTHRDS',16);!Set the number of threads;
  @SET('ITRLIM',72000);!Set the time limit of the model;
  @SET('MXMEMB',1024);!Set Maximum Memory for model;
  ! @MAX/MIN makes model nonlinear.Global solver will linearize @MAX/MIN;
  @SET( 'GLOBAL', 1);
  ! 2. 计算函数对间的通信时间 DELAYFORCU = TENSOR x1 DELAY;
  TensorProduct1;
  
  !@TABLE(FLUXFORFUNCPAIR);
  !@WRITE( AQL);!Deal with var;
  @TEXT() = '  Debug Output Showing Input:';
  !@TABLE( OBJECTCONSTRAINT,2); !Deal with set and attri;
ENDCALC

[OBJ] MIN = MaxRatio;
! Entity Task assignment Mode: Xp Determined by Xe;
${EntityTaskAssignmentMode}

! Workload of each Transport node >=0;
  ! Frobenius inner product x2,3;
${WorkloadOfEachTransportNode}
!@FOR(lv(k):MaxRatio >=
    @SUM(cuxcu(i,j):
        TENSOR(k,i,j)*@SUM(fxf(fi,fj):Xp(i,fi)*FREQUENCY(fi)*FLUXFORFUNCPAIR(fi,fj)*Xp(j,fj)))/BANDWIDTH(k))  ;  !update workload of each Transport node;
! Worklad of each Compute Unit >= 0;
${WorkloadOfEachComputeUnit}
!@FOR(cuv(cui):MaxRatio >=
    @SUM(fv(lfj):Xp(cui,lfj)*WORKLOAD(lfj,lfj))/SPEED(cui));
! The delay of each pair of functions should be less than the constraint;
@FOR(fxf(dfi,dfj):DELAYCONSTRAINT(dfi,dfj)>=
      @SUM(cuxcu(dcui,dcuj):Xp(dcui,dfi)*Xp(dcuj,dfj)*@SUM(lv(k):TENSOR(k,dcui,dcuj)/BANDWIDTH(k)))+
      @SUM(cuxcu(dcui,dcuj):Xp(dcui,dfi)*DELAYFORCU(dcui,dcuj)*Xp(dcuj,dfj)));
! Function assignment matrix Xp should be binary(bool) and satisfy the constraint that only assigned to one Compute Unit;
${XConstraint}


DATA:
 @POINTER( 10) = Xp; ! Function Task assignment matrix(Both);
 @POINTER( 11) = MaxRatio;      ! "Occupy Ratio of computation or communication;
 @POINTER( 12) = @STATUS();
ENDDATA


END
