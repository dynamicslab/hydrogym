n1 = 17.0;
n2 = 5.0;
n3 = 0.7;
x_ninf = -60.0;
x_pinf = 200.0;
y_inf = 20.0;
x_plus = 1.5;

// Origin
Point(1) = {0, 0, 0, 1/n1};

// Cylinder points
Point(2) = { 0.5, 0.0, 0, 1/n1};
Point(3) = { 0.0, 0.5, 0, 1/n1};
Point(4) = {-0.5, 0.0, 0, 1/n1};
Point(5) = { 0.0,-0.5, 0, 1/n1};

// Near boundaries
Point(6) = {-1.5, -1.5, 0, 1/n1};
Point(7) = {-1.5,  1.5, 0, 1/n1};
Point(8) = {x_plus, 1.5, 0, 1/n1};
Point(9) = {x_plus, -1.5, 0, 1/n1};

// Medium boundaries
Point(10) = {x_ninf, -3.0, 0, 1/n2};
Point(11) = {x_ninf, 3.0, 0, 1/n2};
Point(12) = {x_pinf, -3.0, 0, 1/n2};
Point(13) = {x_pinf, 3.0, 0, 1/n2};

// Far boundaries
Point(14) = {x_ninf, -y_inf, 0, 1/n3};
Point(15) = {x_ninf, y_inf, 0, 1/n3};
Point(16) = {x_pinf, -y_inf, 0, 1/n3};
Point(17) = {x_pinf, y_inf, 0, 1/n3};

// Cylinder Lines
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};


//+
Line(5) = {6, 7};
//+
Line(6) = {7, 8};
//+
Line(7) = {8, 9};
//+
Line(8) = {9, 6};
//+
Line(9) = {14, 10};
//+
Line(10) = {10, 11};
//+
Line(11) = {11, 15};
//+
Line(12) = {15, 17};
//+
Line(13) = {17, 13};
//+
Line(14) = {13, 12};
//+
Line(15) = {12, 16};
//+
Line(16) = {16, 14};
//+
Line(17) = {11, 13};
//+
Line(18) = {10, 12};


//+
Curve Loop(1) = {6, 7, 8, 5};
//+
Curve Loop(2) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1, 2};
//+
Curve Loop(3) = {17, 14, -18, 10};
//+
Plane Surface(2) = {1, 3};
//+
Curve Loop(4) = {11, 12, 13, -17};
//+
Plane Surface(3) = {4};
//+
Curve Loop(5) = {9, 18, 15, 16};
//+
Plane Surface(4) = {5};


//+
Physical Surface("Fluid", 1) = {3, 2, 4, 1};
//+
Physical Curve("Inlet", 2) = {11, 10, 9};
//+
Physical Curve("Freestream", 3) = {12, 16};
//+
Physical Curve("Outlet", 4) = {13, 14, 15};
//+
Physical Curve("Wall", 5) = {1, 4, 3, 2};
