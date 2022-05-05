n1 = 10.0;
n2 = 4;
n3 = 1.0;
x_ninf = -6.0;
x_pinf = 20.0;
y_inf = 6.0;

// Cylinder points
R = 0.5;
x0 = 0.0;
y0 = 0.0;
Point(1) = { x0, y0, 0, 1/n1};
Point(2) = { x0+R,y0+0.0, 0, 1/n1};
Point(3) = { x0+0.0,y0+R, 0, 1/n1};
Point(4) = { x0-R,y0+0.0, 0, 1/n1};
Point(5) = { x0+0.0,y0-R, 0, 1/n1};

x0 = 1.5*1.866*R;
y0 = 1.5*R;
Point(6) = { x0, y0, 0, 1/n1};
Point(7) = { x0+R,y0+0.0, 0, 1/n1};
Point(8) = { x0+0.0,y0+R, 0, 1/n1};
Point(9) = { x0-R,y0+0.0, 0, 1/n1};
Point(10) = { x0+0.0,y0-R, 0, 1/n1};

x0 = 1.5*1.866*R;
y0 = -1.5*R;
Point(11) = { x0, y0, 0, 1/n1};
Point(12) = { x0+0.5,y0+0.0, 0, 1/n1};
Point(13) = { x0+0.0,y0+0.5, 0, 1/n1};
Point(14) = { x0-0.5,y0+0.0, 0, 1/n1};
Point(15) = { x0+0.0,y0-0.5, 0, 1/n1};

// Boundaries
Point(16) = {x_ninf, -y_inf, 0, 1/n3};
Point(17) = {x_ninf, 0, 0, 1/n3};
Point(18) = {x_ninf, y_inf, 0, 1/n3};
Point(19) = {x_pinf, -y_inf, 0, 1/n3};
Point(20) = {x_pinf, 0, 0, 1/n2};
Point(21) = {x_pinf, y_inf, 0, 1/n3};

// High-resolution points in near-wake
y0 = 1.5*R;
Point(22) = {5.0,  y0, 0, 1/n1};
Point(23) = {5.0,  -y0, 0, 1/n1};
Point(24) = {x_pinf,  y0, 0, 1/n2};
Point(25) = {x_pinf, -y0, 0, 1/n2};

// Cylinder Lines
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

Circle(5) = {7, 6, 8};
Circle(6) = {8, 6, 9};
Circle(7) = {9, 6, 10};
Circle(8) = {10, 6, 7};

Circle(9)  = {12, 11, 13};
Circle(10) = {13, 11, 14};
Circle(11) = {14, 11, 15};
Circle(12) = {15, 11, 12};

//+
Line(13) = {17, 18};
//+
Line(14) = {18, 21};
//+
Line(15) = {21, 24};
//+
Line(16) = {24, 20};
//+
Line(17) = {20, 25};
//+
Line(18) = {25, 19};
//+
Line(19) = {19, 16};
//+
Line(20) = {16, 17};
//+
Line(21) = {17, 4};
//+
Line(22) = {3, 9};
//+
Line(23) = {7, 22};
//+
Line(24) = {22, 24};
//+
Line(25) = {5, 14};
//+
Line(26) = {12, 23};
//+
Line(27) = {23, 25};
//+
Curve Loop(1) = {13, 14, 15, -24, -23, 5, 6, -22, 2, -21};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {21, 3, 25, 11, 12, 26, 27, 18, 19, 20};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {22, 7, 8, 23, 24, 16, 17, -27, -26, 9, 10, -25, 4, 1};
//+
Plane Surface(3) = {3};
//+
Physical Surface("Fluid", 1) = {1, 3, 2};
//+
Physical Curve("Inlet", 2) = {13, 20};
//+
Physical Curve("Freestream", 3) = {14, 19};
//+
Physical Curve("Outlet", 4) = {15, 16, 17, 18};
//+
Physical Curve("Cyl1", 5) = {2, 1, 4, 3};
//+
Physical Curve("Cyl2", 6) = {6, 5, 8, 7};
//+
Physical Curve("Cyl3", 7) = {10, 9, 12, 11};
