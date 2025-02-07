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
! Make all the vectors, column vectors;
  lxcxc( lk, cu, cu):  TENSOR;
  fxf(f, f): OMEGA,WORKLOAD,DELAYCONSTRAINT,FLUXFORFUNCPAIR,OBJECTCONSTRAINT;
  lv(lk): BANDWIDTH, DELAY,lkload;
  cuv(cu): SPEED,cuload;
  fv(f): fvDemo;
  cuxcu(cu, cu): DELAYFORCU;
! objective function;
  cuxf(cu, f): Xp;
ENDSETS

DATA:
  TENSOR  = @POINTER( 1);     ! "Topology" for different Compute Units with Transport nodes;
  FLUXFORFUNCPAIR = @POINTER( 2);    ! "Data Load" of states;
  WORKLOAD = @POINTER( 3);        ! "Workload" of f diag([workload,workload,...,workload]);
  BANDWIDTH = @POINTER( 4);       ! "Bandwidth" of Transport nodes;
  SPEED = @POINTER( 5);           ! "Speed" of Compute Units;
  DELAY = @POINTER( 6);           ! "Delay" of Transport nodes;
  DELAYCONSTRAINT = @POINTER( 7); ! "Delay" constraint for pair(fi,fj);
  OBJECTCONSTRAINT = @POINTER( 8); ! Bool flag for "Object" constraint for pair(fi,fj);
ENDDATA


procedure TensorProduct1:
! Frobenius inner product x1;
  @FOR(cuxcu(i,j):DELAYFORCU(i,j)=
       @SUM(lv(k):
            TENSOR(k,i,j)*DELAY(k)))  ;
endprocedure

CALC:
  ! 2. 计算函数对间的通信时间 DELAYFORCU = TENSOR x1 DELAY;
  TensorProduct1;
  
  !@TABLE(FLUXFORFUNCPAIR);
  !@WRITE( AQL);!Deal with var;
  @TEXT() = '  Debug Output Showing Input:';
  @TABLE( DELAYFORCU,2); !Deal with set and attri;
ENDCALC

[OBJ] MIN = MaxRatio;
! Workload of each Transport node >=0;
  ! Frobenius inner product x2,3;
@FOR(lv(k):MaxRatio >= 
    @SUM(cuxcu(i,j):
        TENSOR(k,i,j)*@SUM(fxf(fi,fj):Xp(i,fi)*FLUXFORFUNCPAIR(fi,fj)*Xp(j,fj)))/BANDWIDTH(k))  ;  !update workload of each Transport node;
! Worklad of each Compute Unit >= 0;
@FOR(cuv(cui):MaxRatio >=
    @SUM(fv(lfj):Xp(cui,lfj)*WORKLOAD(lfj,lfj))/SPEED(cui));
! The delay of each pair of functions should be less than the constraint;
@FOR(fxf(dfi,dfj):DELAYCONSTRAINT(dfi,dfj)>=@SUM(cuxcu(dcui,dcuj):Xp(dcui,dfi)*DELAYFORCU(dcui,dcuj)*Xp(dcuj,dfj)));
! Function assignment matrix Xp should be binary(bool) and satisfy the constraint that only assigned to one Compute Unit;
@FOR(fv(k):@SUM(cuv(ci):Xp(ci,k))=1);
@FOR(cuxf(i,j):@BIN(Xp(i,j)));
!Xp(4,12)=1; !强制要求某个函数在对应CU位置@INDEX(Xp,12)=1 wrong use of index;
!@FOR(cuv(i):
	@FOR(fv(j):
  	 @SUM(fv(k): Xp(i,k)*OBJECTCONSTRAINT(j,k))>=@SUM(fv(k):OBJECTCONSTRAINT(j,k)))); !DEAL WITH F assigned Constraint;


DATA:
 @POINTER( 9) = Xp;            ! "Task assignment matrix";
 @POINTER( 10) = MaxRatio;      ! "Occupy Ratio of computation or communication;
 @POINTER( 11) = @STATUS();
ENDDATA


END
