n = 100;
n1p = 50;
n2p = 25;
n3p = 15;
n1m = 40;
n2m = 25;
n3m = 15;

// Thick solid lines
Point(1) = { -0.6, 0.0, 0, 1/n};
Point(2) = { -0.6, 0.1, 0, 1/n};
Point(3) = {  2.5, 0.1, 0, 1/n};
Point(4) = {  2.5, 0.0, 0, 1/n};
Point(5) = { 1.75, 0.0, 0, 1/n};
Point(6) = {  1.0, 0.0, 0, 1/n};
Point(7) = {  1.0,-0.1, 0, 1/n};
Point(8) = {  0.0,-0.1, 0, 1/n};
Point(9) = {  0.0, 0.0, 0, 1/n};
Point(10) = { -0.4, 0.0, 0, 1/n};

// Thin solid lines
Point(11) = {-1.2, 0.0, 0, 1/n1p};
Point(12) = {-1.2,0.15, 0, 1/n1p};
Point(13) = { 2.5,0.15, 0, 1/n1p};
Point(14) = { 1.0,-0.2, 0, 1/n1m};
Point(15) = { 0.0,-0.2, 0, 1/n1m};

// Dashed lines
Point(16) = {-1.2, 0.3, 0, 1/n2p};
Point(17) = { 2.5, 0.3, 0, 1/n2p};
Point(18) = { 1.0,-0.35, 0, 1/n2m};
Point(19) = { 0.0,-0.35, 0, 1/n2m};

// Dotted lines
Point(20) = {-1.2, 0.5, 0, 1/n3p};
Point(21) = { 2.5, 0.5, 0, 1/n3p};
Point(22) = { 1.0,-1.0, 0, 1/n3m};
Point(23) = { 0.0,-1.0, 0, 1/n3m};

// Control location
Point(24) = {-0.35, 0.0, 0, 1/n};

// Measurement location
Point(25) = { 1.1, 0.0, 0, 1/n};

//+
Line(1) = {11, 12};
//+
Line(2) = {12, 16};
//+
Line(3) = {16, 20};
//+
Line(4) = {20, 21};
//+
Line(5) = {16, 17};
//+
Line(6) = {12, 13};
//+
Line(7) = {11, 1};
//+
Line(8) = {1, 10};
//+
Line(9) = {10, 24};
//+
Line(10) = {9, 6};
//+
Line(11) = {6, 25};
//+
Line(12) = {5, 4};
//+
Line(13) = {21, 17};
//+
Line(14) = {17, 13};
//+
Line(15) = {13, 3};
//+
Line(16) = {3, 4};
//+
Line(17) = {2, 3};
//+
Line(18) = {1, 2};
//+
Line(19) = {9, 8};
//+
Line(20) = {8, 15};
//+
Line(21) = {15, 19};
//+
Line(22) = {19, 23};
//+
Line(23) = {23, 22};
//+
Line(24) = {22, 18};
//+
Line(25) = {18, 14};
//+
Line(26) = {14, 7};
//+
Line(27) = {7, 6};
//+
Line(28) = {8, 7};
//+
Line(29) = {15, 14};
//+
Line(30) = {19, 18};
//+
Line(31) = {24, 9};
//+
Line(32) = {25, 5};

//+
Curve Loop(1) = {18, 17, 16, -12, -11, -32, -27, -28, -19, -31, -9, -8};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {1, 6, 15, -17, -18, -7};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {2, 5, 14, -6};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {3, 4, 13, -5};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {28, -26, -29, -20};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {29, -25, -30, -21};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {30, -24, -23, -22};
//+
Plane Surface(7) = {7};
// 
Physical Surface("Fluid", 1) = {4, 3, 2, 1, 5, 6, 7};
//+
Physical Curve("Inlet", 2) = {3, 2, 1};
//+
Physical Curve("Freestream", 3) = {4};
//+
Physical Curve("Outlet", 4) = {13, 14, 15, 16};
//  Note that the downstream wall is "12"... move that up here to add that to the slip BCs
Physical Curve("Slip", 5) = {7, 8};
//+
Physical Curve("Wall", 6) = {9, 19, 20, 21, 22, 23, 24, 25, 26, 27, 32, 12};
//+
Physical Curve("Control", 7) = {31};
//+
Physical Curve("Sensor", 8) = {11};
