n1 = 35.0;
n2 = 20.0;
n3 = 1.5;
x_ninf = -5.0;
x_pinf = 15.0;
y_inf = 5.0;

// Origin
Point(1) = {0, 0, 0, 1/n1};

// Cylinder points
Point(2) = { 0.5, 0.0, 0, 1/n1};
Point(3) = { 0.0, 0.5, 0, 1/n1};
Point(4) = {-0.5, 0.0, 0, 1/n1};
Point(5) = { 0.0,-0.5, 0, 1/n1};

// Boundaries
Point(6) = {x_ninf, -y_inf, 0, 1/n3};
Point(7) = {x_ninf, 0, 0, 1/n3};
Point(8) = {x_ninf, y_inf, 0, 1/n3};
Point(9) = {x_pinf, -y_inf, 0, 1/n3};
Point(10) = {x_pinf, 0, 0, 1/n2};
Point(11) = {x_pinf, y_inf, 0, 1/n3};

// Actuation limits
Point(12) = { 2.0,  0,  0, 1/n2};
Point(13) = { 2.5, 0.0, 0, 1/n2};
Point(14) = { 2.0, 0.5, 0, 1/n2};
Point(15) = { 1.5, 0.0, 0, 1/n2};
Point(16) = { 2.0,-0.5, 0, 1/n2};

// Cylinder Lines
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

// Boundary lines
Line(5) = {6, 7};
Line(6) = {7, 8};
Line(7) = {8, 11};
Line(8) = {11, 10};
Line(9) = {10, 9};
Line(10) = {9, 6};
Line(11) = {7, 4};
Line(12) = {2, 15};

// Actuation disk
Circle(13) = {13, 12, 14};
Circle(14) = {14, 12, 15};
Circle(15) = {15, 12, 16};
Circle(16) = {16, 12, 13};
Line(17) = {13, 10};

//+
Curve Loop(1) = {7, 8, -17, 13, 14, -12, 1, 2, -11, 6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {11, 3, 4, 12, 15, 16, 17, 9, 10, 5};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {13, 14, 15, 16};
//+
Plane Surface(3) = {3};

//+
Physical Surface("Fluid", 1) = {1, 2};
//+
Physical Curve("Inlet", 2) = {6, 5};
//+
Physical Curve("Freestream", 3) = {7, 10};
//+
Physical Curve("Outlet", 4) = {8, 9};
//+
Physical Curve("Wall", 5) = {1, 4, 3, 2};

Physical Surface("Actuation", 6) = {3};