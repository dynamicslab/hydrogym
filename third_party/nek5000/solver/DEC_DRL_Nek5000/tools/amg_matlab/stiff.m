function A = stiff(e)
  global conn vert
  v = vert(conn(e,:),:)';
  D = zeros(2,4);
  D(:,1) = v(:,2)-v(:,1);
  D(:,2) = v(:,4)-v(:,3);
  D(:,3) = v(:,3)-v(:,1);
  D(:,4) = v(:,4)-v(:,2);
  A =( st(D,0,0)+st(D,1,0)+st(D,0,1)+st(D,1,1) ...
      +4*(st(D,0,.5)+st(D,1,.5)+st(D,.5,0)+st(D,.5,1)) ...
	    +16*st(D,.5,.5))/36;
end
function A = st(D,r,s)
  dxdr = [(1-s)*D(:,1)+s*D(:,2)   (1-r)*D(:,3)+r*D(:,4)];
  drdx = inv(dxdr);
  jac = abs(det(dxdr));
  dhdr = [ -(1-s), -(1-r);
            (1-s), -r;
	  				-s   ,  (1-r);
		  			 s   ,  r ];
  dhdx = dhdr*drdx;
  A = dhdx * jac * dhdx';
end
