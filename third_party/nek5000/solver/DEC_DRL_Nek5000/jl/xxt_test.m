function xxt_test

Al0=[8 -1; -1 4];
Ac0=[-2 -2 -2; 0 -2 -1];
As0=[4 -1 0; -1 8 -1; 0 -1 4];

Al1=[4];
Ac1=[-1 -1 -2];
As1=[4 -2 -1; -2 4 -1; -1 -1 4];

Al2=[4];
Ac2=[-1 -2 -1];
As2=[4 -1 -2; -1 4 -1; -2 -1 4];

A0=[Al0 Ac0; Ac0' As0];
A1=[Al1 Ac1; Ac1' As1];
A2=[Al2 Ac2; Ac2' As2];

Il=eye(4); Is=eye(4);
gI=eye(8);
Rl0=Il([1 2],:);
Rl1=Il([3],:);
Rl2=Il([4],:);
Rs0=Is([6 7 8]-4,:);
Rs1=Is([5 6 7]-4,:);
Rs2=Is([5 7 8]-4,:);
Zls0 = zeros(length(Rl0(:,1)),length(Rs0(1,:)));
Zls1 = zeros(length(Rl1(:,1)),length(Rs1(1,:)));
Zls2 = zeros(length(Rl2(:,1)),length(Rs2(1,:)));
Zsl0 = zeros(length(Rs0(:,1)),length(Rl0(1,:)));
Zsl1 = zeros(length(Rs1(:,1)),length(Rl1(1,:)));
Zsl2 = zeros(length(Rs2(:,1)),length(Rl2(1,:)));
R0=[Rl0 Zls0; Zsl0 Rs0];
R1=[Rl1 Zls1; Zsl1 Rs1];
R2=[Rl2 Zls2; Zsl2 Rs2];

A=R0'*A0*R0+R1'*A1*R1+R2'*A2*R2;

All = bdiag(Al0,Al1,Al2);
Als = bdiag(Ac0,Ac1,Ac2);
Ass = bdiag(As0,As1,As2);
Qs = [Rs0;Rs1;Rs2];
ns = length(Qs(:,1));

Q = bdiag(eye(4),Qs,[]);

A=[All Als*Qs; Qs'*Als' Qs'*Ass*Qs];

M = Q*(A\Q');

S0 = As0-Ac0'*(Al0\Ac0);
S1 = As1-Ac1'*(Al1\Ac1);
S2 = As2-Ac2'*(Al2\Ac2);

dS = bdiag(S0,S1,S2);

em=[eye(4) -All\Als; 0*Als' eye(ns)];
norm(M-em*[inv(All) 0*Als; 0*Als' Qs*inv(Qs'*dS*Qs)*Qs']*em');

Rf0=Is([6 7 8]-4,:);
Rf1=Is([5 6 7 8]-4,:);
Rf2=Is([5 6 7 8]-4,:);
Qf = [Rf0;Rf1;Rf2];

X0 = zeros(length(Rs0(:,1)),length(Rf0(:,1)));
X1 = zeros(length(Rs1(:,1)),length(Rf1(:,1)));
X2 = zeros(length(Rs2(:,1)),length(Rf2(:,1)));
dX = bdiag(X0,X1,X2);
X = Qs'*dX*Qf;

X0 = Rs0*X*Rf0';

n = length(Qs(1,:)); Is = eye(n);
for i = [1:n]
  ei = Is(:,i);
	Ri = Is([1:i-1],:);
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
  X2 = Rs2*X*Rf2'
	dX = bdiag(X0,X1,X2);
	pause
end

norm(M-em*[inv(All) 0*Als; 0*Als' Qs*X*X'*Qs']*em')
norm(M-em*[inv(All) 0*Als; 0*Als' dX*Qf*Qf'*dX']*em')

X

inv(chol(A))

Ai=inv(A);

Ai([6 1 7 1 7 2 8],[6 3 1 7 5 2 8 4])'
Ai([6 3 7 5],[6 3 1 7 5 2 8 4])'
Ai([7 5 8 4],[6 3 1 7 5 2 8 4])'

end

function M=bdiag(A,B,C)
	[ra ca]=size(A);
	[rb cb]=size(B);
	[rc cc]=size(C);
  M = [ A            zeros(ra,cb) zeros(ra,cc)
	      zeros(rb,ca) B            zeros(rb,cc)
				zeros(rc,ca) zeros(rc,cb) C ];
end	
