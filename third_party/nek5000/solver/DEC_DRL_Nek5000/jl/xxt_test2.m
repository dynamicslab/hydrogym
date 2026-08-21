%p = [4 3 2 1 3 6 1 5 6  5 ]

%inv(A)(p,p)

function M=bdiag(A,B,C)
	[ra ca]=size(A);
	[rb cb]=size(B);
	[rc cc]=size(C);
  M = [ A            zeros(ra,cb) zeros(ra,cc)
	      zeros(rb,ca) B            zeros(rb,cc)
				zeros(rc,ca) zeros(rc,cb) C ];
end	

Al0=[];
Ac0=zeros(2)([],:);
As0=[1 -.5; -.5 1];

Al1=[  2 -.5  -1   0
     -.5   1   0 -.5
		  -1   0   2 -.5
			 0 -.5 -.5   1];
Ac1=[-.5 0; 0 0; 0 -.5; 0 0];
As1=[1 -.5; -.5 1];

A0=[Al0 Ac0; Ac0' As0];
A1=[Al1 Ac1; Ac1' As1];

Il=eye(4); Is=eye(2);
gI=eye(6);
Rl0=Il([],:);
Rl1=Il([1 2 3 4],:);
Rs0=Is([5 6]-4,:);
Rs1=Is([5 6]-4,:);
R0=[Rl0 zeros(size(Rl0)(1),size(Rs0)(2))
    zeros(size(Rs0)(1),size(Rl0)(2)) Rs0];
R1=[Rl1 zeros(size(Rl1)(1),size(Rs1)(2))
    zeros(size(Rs1)(1),size(Rl1)(2)) Rs1];

A=R0'*A0*R0+R1'*A1*R1;

All = bdiag(Al0,Al1,[])
Als = bdiag(Ac0,Ac1,[])
Ass = bdiag(As0,As1,[])
Qs = [Rs0;Rs1]
ns = size(Qs)(1);

Q = bdiag(Il,Qs,[]);

A=[All Als*Qs; Qs'*Als' Qs'*Ass*Qs];

M = Q*(A\Q');

S0 = As0;
S1 = As1-Ac1'*(Al1\Ac1);

dS = bdiag(S0,S1,[]);

em=[Il -All\Als; 0*Als' eye(ns)];
norm(M-em*[inv(All) 0*Als; 0*Als' Qs*inv(Qs'*dS*Qs)*Qs']*em')

Rf0=Is([5 6]-4,:);
Rf1=Is([5 6]-4,:);
Qf = [Rf0;Rf1];

X0 = zeros(size(Rs0)(1),size(Rf0)(1));
X1 = zeros(size(Rs1)(1),size(Rf1)(1));
dX = bdiag(X0,X1,[]);
X = Qs'*dX*Qf;

X0 = Rs0*X*Rf0';

n = size(Qs)(2)
for i = [1:n]
  ei = eye(n)(:,i);
	Ri = eye(n)([1:i-1],:);
  se = dS*Qs*ei
	Xtse = dX'*se
	QQtXtse = Qf*Ri'*Ri*Qf'*Xtse
	Qy = Qs*ei - dX*QQtXtse
	ytsy = Qy'*dS*Qy
	Qx = Qy/sqrt(ytsy)
	xv = inv(Qs'*Qs)*Qs'*Qx;
	X(:,i)=xv
  X0 = Rs0*X*Rf0'
  X1 = Rs1*X*Rf1'
	dX = bdiag(X0,X1,[]);
	pause
end

norm(M-em*[inv(All) 0*Als; 0*Als' Qs*X*X'*Qs']*em')
norm(M-em*[inv(All) 0*Als; 0*Als' dX*Qf*Qf'*dX']*em')

X

inv(chol(A))


