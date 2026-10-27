      BLOCKDATA EDGEC
C
C     Note, for some (lesser) compilers, the BLOCKDATA statement
C     needs to be compiled in a separate file.  Therefore, do not
C     include any other modules in this file.
C
      INCLUDE 'SIZE'
      INCLUDE 'TOPOL'
      DATA    IEDGFC /  5,7,9,11,  6,8,10,12,   
     $                  1,3,9,10,  2,4,11,12,
     $                  1,2,5,6,   3,4,7,8    /
      DATA    ICEDG / 1,2,1,   3,4,1,   5,6,1,   7,8,1,
     $                1,3,2,   2,4,2,   5,7,2,   6,8,2,
     $                1,5,3,   2,6,3,   3,7,3,   4,8,3,
C      -2D-
     $                1,2,1,   3,4,1,   1,3,2,   2,4,2 /
      DATA    ICFACE/ 1,3,5,7, 2,4,6,8,
     $                1,2,5,6, 3,4,7,8,
     $                1,2,3,4, 5,6,7,8,
C      -2D-
     $                1,3,0,0, 2,4,0,0,
     $                1,2,0,0, 3,4,0,0  /
C
      END
