n = 200;   // Near corner
n1 = 150;  // Near walls
n2 = 30;   // Bulk flow

L_in = 2.5;
L_out = 25;
h_in = 0.5;
h_s  = 0.5;
x_lr = 5.9;  // Approximate reattachment point (Boujo & Gallaire, 2015)
dx_s = 0.2;   // Length of sensor region

Point(1) = {-L_in,  0.0, 0, 1/n1};
Point(2) = {-L_in, h_in, 0, 1/n1};
Point(3) = {  0.0, h_in, 0, 1/n1};
Point(4) = {  0.0,  0.0, 0, 1/n};
Point(5) = {  0.0, -h_s, 0, 1/n1};
Point(6) = {L_out, h_in, 0, 1/n1};
Point(7) = {L_out,  0.0, 0, 1/n2};
Point(8) = {L_out, -h_s, 0, 1/n1};

Point(9)  = { -0.5, 0.0, 0, 1/n};
Point(10) = { -0.5, 0.2, 0, 1/n};
Point(11) = {  0.5, 0.2-h_s, 0, 1/n};
Point(12) = {  0.5, 0.2, 0, 1/n};
Point(13) = {  0.5, 0.2-h_s, 0, 1/n};
Point(14) = {  0.0, 0.2-h_s, 0, 1/n};

Point(15) = {-L_in, 0.2, 0, 1/n2};
Point(16) = { -1.0, 0.2, 0, 1/n2};
Point(17) = {  2.0, 0.2, 0, 1/n2};
Point(18) = {  2.0, 0.2-h_s, 0, 1/n2};
Point(19) = {L_out, 0.2-h_s, 0, 1/n2};

Point(20) = {-L_in, h_in-0.2, 0, 1/n2};
Point(21) = { -1.0, h_in-0.2, 0, 1/n2};
Point(22) = {  2.0, h_in-0.2, 0, 1/n2};
Point(23) = {L_out, h_in-0.2, 0, 1/n2};

// Control location
Point(24) = {-0.35, 0.0, 0, 1/n};

// Sensor location
Point(25) = {x_lr, -h_s, 0.0, 1/n1};
Point(26) = {x_lr+dx_s, -h_s, 0.0, 1/n1};

//+
Line(1) = {1, 15};
//+
Line(2) = {15, 20};
//+
Line(3) = {20, 2};
//+
Line(4) = {2, 3};
//+
Line(5) = {1, 9};
//+
Line(6) = {15, 16};
//+
Line(7) = {20, 21};
//+
Line(8) = {21, 22};
//+
Line(9) = {16, 10};
//+
Line(10) = {10, 12};
//+
Line(11) = {12, 17};
//+
Line(12) = {10, 9};
//+
Line(13) = {9, 24};
//+
Line(14) = {4, 14};
//+
Line(15) = {14, 5};
//+
Line(16) = {14, 11};
//+
Line(17) = {11, 12};
//+
Line(18) = {11, 18};
//+
Line(19) = {18, 17};
//+
Line(20) = {17, 22};
//+
Line(21) = {5, 25};
//+
Line(22) = {8, 19};
//+
Line(23) = {19, 7};
//+
Line(24) = {7, 23};
//+
Line(25) = {23, 6};
//+
Line(26) = {19, 18};
//+
Line(27) = {22, 23};
//+
Line(28) = {6, 3};
//+
Line(29) = {24, 4};
//+
Line(30) = {25, 26};
//+
Line(31) = {26, 8};
//+
Curve Loop(1) = {4, -28, -25, -27, -8, -7, 3};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {7, 8, -20, -11, -10, -9, -6, 2};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {6, 9, 12, -5, 1};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {10, -17, -16, -14, -13, -29, -12};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {11, -19, -18, 17};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {18, -26, -22, -21, -30, -31, -15, 16};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {27, -24, -23, 26, 19, 20};
//+
Plane Surface(7) = {7};
//+
Physical Surface("Fluid", 1) = {7, 1, 6, 3, 2, 4, 5};
//+
Physical Curve("Inlet", 2) = {1, 2, 3};
//+
Physical Curve("Outlet", 3) = {25, 24, 23, 22};
//+
Physical Curve("Wall", 4) = {4, 28, 21, 31, 15, 14, 13, 5};
//+
Physical Curve("Control", 5) = {29};
//+
Physical Curve("Sensor", 6) = {30};