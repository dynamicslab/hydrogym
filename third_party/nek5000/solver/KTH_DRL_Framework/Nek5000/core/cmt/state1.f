!*********************************************************************
!* Illinois Open Source License                                      *
!*                                                                   *
!* University of Illinois/NCSA                                       * 
!* Open Source License                                               *
!*                                                                   *
!* Copyright@2008, University of Illinois.  All rights reserved.     *
!*                                                                   *
!*  Developed by:                                                    *
!*                                                                   *
!*     Center for Simulation of Advanced Rockets                     *
!*                                                                   *
!*     University of Illinois                                        *
!*                                                                   *
!*     www.csar.uiuc.edu                                             *
!*                                                                   *
!* Permission is hereby granted, free of charge, to any person       *
!* obtaining a copy of this software and associated documentation    *
!* files (the "Software"), to deal with the Software without         *
!* restriction, including without limitation the rights to use,      *
!* copy, modify, merge, publish, distribute, sublicense, and/or      *
!* sell copies of the Software, and to permit persons to whom the    *
!* Software is furnished to do so, subject to the following          *
!* conditions:                                                       *
!*                                                                   *
!*                                                                   *
!* @ Redistributions of source code must retain the above copyright  * 
!*   notice, this list of conditions and the following disclaimers.  *
!*                                                                   * 
!* @ Redistributions in binary form must reproduce the above         *
!*   copyright notice, this list of conditions and the following     *
!*   disclaimers in the documentation and/or other materials         *
!*   provided with the distribution.                                 *
!*                                                                   *
!* @ Neither the names of the Center for Simulation of Advanced      *
!*   Rockets, the University of Illinois, nor the names of its       *
!*   contributors may be used to endorse or promote products derived * 
!*   from this Software without specific prior written permission.   *
!*                                                                   *
!* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,   *
!* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   *
!* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          *
!* NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR           *
!* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER       * 
!* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,   *
!* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE    *
!* USE OR OTHER DEALINGS WITH THE SOFTWARE.                          *
!*********************************************************************
!* Please acknowledge The University of Illinois Center for          *
!* Simulation of Advanced Rockets in works and publications          *
!* resulting from this software or its derivatives.                  *
!*********************************************************************
!******************************************************************************
!
! Purpose: Collect relations for static and total speed of sound for perfect
!   gases.
!
! Description: None.
!
! Notes: None.
!
!******************************************************************************
!
! $Id: MixtPerf.f,v 1.5 2015/07/17 15:58:14 mrugeshs Exp $
!
! Copyright: (c) 2002 by the University of Illinois
!
!******************************************************************************

      FUNCTION MixtJWL_Enthalpy(DE,PRES,VEL1,VEL2,VEL3,EN)
      IMPLICIT NONE
      REAL DE,PRES,VEL1,VEL2,VEL3,EN
      REAL MixtJWL_Enthalpy
      MixtJWL_Enthalpy = 0.5*(VEL1*VEL1+VEL2*VEL2+VEL3*VEL3)
     >                  + EN+PRES/DE
      END
      FUNCTION MixtPerf_C_Co2GUVW(Co2,G,U,V,W)
      IMPLICIT NONE
      REAL Co2,G,U,V,W ! INTENT(IN) W
      REAL MixtPerf_C_Co2GUVW
      MixtPerf_C_Co2GUVW = SQRT(Co2 - 0.5*(G - 1.0)*(U*U + V*V + W*W))
      END

      FUNCTION MixtPerf_C_DGP(D,G,P)
      IMPLICIT NONE
      REAL D,G,P! INTENT(IN) 
      REAL MixtPerf_C_DGP
      MixtPerf_C_DGP = SQRT(G*P/D)
      END

      FUNCTION MixtPerf_C_GHoVm2(G,Ho,Vm2)
      IMPLICIT NONE
      REAL G,Ho,Vm2! INTENT(IN) 
      REAL MixtPerf_C_GHoVm2
      MixtPerf_C_GHoVm2 = SQRT((G - 1.0)*(Ho - 0.5*Vm2))
      END

      FUNCTION MixtPerf_C_GRT(G,R,T)
      IMPLICIT NONE
      REAL G,R,T! INTENT(IN) 
      REAL MixtPerf_C_GRT
      MixtPerf_C_GRT = SQRT(G*R*T)
      END

      FUNCTION MixtPerf_C2_GRT(G,R,T)
      IMPLICIT NONE
      REAL G,R,T! INTENT(IN) 
      REAL MixtPerf_C2_GRT
      MixtPerf_C2_GRT = G*R*T
      END

      FUNCTION MixtPerf_Co2_CGUVW(C,G,U,V,W)
      IMPLICIT NONE
      REAL C,G,U,V,W! INTENT(IN) 
      REAL MixtPerf_Co2_CGUVW
      MixtPerf_Co2_CGUVW = C*C + 0.5*(G - 1.0)*(U*U + V*V + W*W)
      END

      FUNCTION MixtPerf_Cv_CpR(Cp,R)
      IMPLICIT NONE
      REAL Cp,R! INTENT(IN) 
      REAL MixtPerf_Cv_CpR
      MixtPerf_Cv_CpR = Cp - R  
      END

      FUNCTION MixtPerf_Cp_CvR(Cv,R)
      IMPLICIT NONE
      REAL Cv,R! INTENT(IN) 
      REAL MixtPerf_Cp_CvR
      MixtPerf_Cp_CvR = Cv + R  
      END

      FUNCTION MixtPerf_D_CGP(C,G,P)
      IMPLICIT NONE
      REAL C,G,P! INTENT(IN) 
      REAL MixtPerf_D_CGP
      MixtPerf_D_CGP = G*P/(C*C)
      END

      FUNCTION MixtPerf_D_DoGMa(D,G,Ma)
      IMPLICIT NONE
      REAL D,G,Ma! INTENT(IN) 
      REAL MixtPerf_D_DoGMa
      MixtPerf_D_DoGMa = D/ (1.0 + 0.5*(G-1.0)*Ma*Ma)**(1.0/(G-1.0))
      END

      FUNCTION MixtPerf_D_PRT(P,R,T)
      IMPLICIT NONE
      REAL P,R,T ! INTENT(IN) 
      REAL MixtPerf_D_PRT
      MixtPerf_D_PRT = P/(R*T)
      END

      FUNCTION MixtPerf_Eo_DGPUVW(D,G,P,U,V,W)
      IMPLICIT NONE
      REAL D,G,P,U,V,W! INTENT(IN) 
      REAL MixtPerf_Eo_DGPUVW
      MixtPerf_Eo_DGPUVW = P/(D*(G - 1.0)) + 0.5*(U*U + V*V + W*W)
      END

      FUNCTION MixtPerf_Eo_DGPVm(D,G,P,Vm)
      IMPLICIT NONE
      REAL D,G,P,Vm! INTENT(IN) 
      REAL MixtPerf_Eo_DGPVm
      MixtPerf_Eo_DGPVm = P/(D*(G - 1.0)) + 0.5*Vm*Vm
      END

      FUNCTION MixtPerf_Eo_GRTUVW(G,R,T,U,V,W)
      IMPLICIT NONE
      REAL  G,R,T,U,V,W
      REAL  MixtPerf_Eo_GRTUVW
      MixtPerf_Eo_GRTUVW = R*T/(G - 1.0) + 0.5*(U*U + V*V + W*W)
      END

      FUNCTION MixtPerf_G_CpR(Cp,R)
      IMPLICIT NONE
      REAL  Cp,R
      REAL  MixtPerf_G_CpR
      MixtPerf_G_CpR = Cp/(Cp - R)
      END

      FUNCTION MixtPerf_Ho_CpTUVW(Cp,T,U,V,W)
      IMPLICIT NONE
      REAL  Cp,T,U,V,W
      REAL  MixtPerf_Ho_CpTUVW
      MixtPerf_Ho_CpTUVW = Cp*T + 0.5*(U*U + V*V + W*W)
      END

      FUNCTION MixtPerf_M_R(R)
      IMPLICIT NONE
      REAL  R
      REAL  MixtPerf_M_R
      MixtPerf_M_R = 8314.3/R
      END

      FUNCTION MixtPerf_P_DEoGVm2(D,Eo,G,Vm2)
      IMPLICIT NONE
      REAL  D,Eo,G,Vm2
      REAL  MixtPerf_P_DEoGVm2
      MixtPerf_P_DEoGVm2 = (G - 1.0)*D*(Eo - 0.5*Vm2)
      END

      FUNCTION MixtPerf_P_DRT(D,R,T)
      IMPLICIT NONE
      REAL  D,R,T
      REAL  MixtPerf_P_DRT
      MixtPerf_P_DRT = D*R*T
      END

      FUNCTION MixtPerf_P_GMaPo(G,Ma,Po)
      IMPLICIT NONE
      REAL  G,Ma,Po
      REAL  MixtPerf_P_GMaPo
      MixtPerf_P_GMaPo = Po/((1.0 + 0.5*(G - 1.0)*Ma*Ma)**(G/(G - 1.0)))
      END

      FUNCTION MixtPerf_P_DDoGPo(D,Do,G,Po)
      IMPLICIT NONE
      REAL  D,Do,G,Po
      REAL  MixtPerf_P_DDoGPo
      MixtPerf_P_DDoGPo = Po*(D/Do)**G
      END

      FUNCTION MixtPerf_P_GPoTTo(G,Po,T,To)
      IMPLICIT NONE
      REAL  G,Po,T,To
      REAL  MixtPerf_P_GPoTTo
      MixtPerf_P_GPoTTo = Po*(T/To)**(G/(G - 1.0))
      END

      FUNCTION MixtPerf_Po_GPTTo(G,P,T,To)
      IMPLICIT NONE
      REAL  G,P,T,To
      REAL  MixtPerf_Po_GPTTo
      MixtPerf_Po_GPTTo = P*(To/T)**(G/(G - 1.0))
      END

      FUNCTION MixtPerf_Po_CGPUVW(C,G,P,U,V,W)
      IMPLICIT NONE
      REAL  C,G,P,U,V,W
      REAL  MixtPerf_Po_CGPUVW
      MixtPerf_Po_CGPUVW =
     >        P*(1.0 + 0.5*(G - 1.0)*(U*U+V*V+W*W)/(C*C))**(G/(G - 1.0))
      END

      FUNCTION MixtPerf_R_CpG(Cp,G)
      IMPLICIT NONE
      REAL  Cp,G
      REAL  MixtPerf_R_CpG
      MixtPerf_R_CpG = Cp - Cp/G  
      END

      FUNCTION MixtPerf_R_M(M,whatev)
      IMPLICIT NONE
      REAL  M,whatev
      REAL  MixtPerf_R_M
      MixtPerf_R_M = 8314.3/M
      END

      FUNCTION MixtPerf_T_CGR(C,G,R)
      IMPLICIT NONE
      REAL  C,G,R
      REAL  MixtPerf_T_CGR
      MixtPerf_T_CGR = C*C/(G*R)
      END

      FUNCTION MixtPerf_T_CpHoVm2(Cp,Ho,Vm2)
      IMPLICIT NONE
      REAL  Cp,Ho,Vm2
      REAL  MixtPerf_T_CpHoVm2
      MixtPerf_T_CpHoVm2 = (Ho-0.5*Vm2)/Cp
      END

      FUNCTION MixtPerf_T_CvEoVm2(Cv,Eo,Vm2)
      IMPLICIT NONE
      REAL  Cv,Eo,Vm2
      REAL  MixtPerf_T_CvEoVm2
      MixtPerf_T_CvEoVm2 = (Eo-0.5*Vm2)/Cv
      END

      FUNCTION MixtPerf_T_DPR(D,P,R)
      IMPLICIT NONE
      REAL  D,P,R
      REAL  MixtPerf_T_DPR
      MixtPerf_T_DPR = P/(D*R)
      END

      FUNCTION MixtPerf_HM_T_DGMP(D,G,M,P)
      IMPLICIT NONE
      REAL  D,G,M,P
      REAL  MixtPerf_HM_T_DGMP
      MixtPerf_HM_T_DGMP = (G*M*M*P + 1.0)/D 
      END

      FUNCTION MixtPerf_T_GMaTo(G,Ma,To)
      IMPLICIT NONE
      REAL  G,Ma,To
      REAL  MixtPerf_T_GMaTo
      MixtPerf_T_GMaTo = To/(1.0 + 0.5*(G - 1.0)*Ma*Ma)
      END

      FUNCTION MixtPerf_To_CpTUVW(Cp,T,U,V,W)
      IMPLICIT NONE
      REAL  Cp,T,U,V,W
      REAL  MixtPerf_To_CpTUVW
      MixtPerf_To_CpTUVW = T + 0.5*(U*U + V*V + W*W)/Cp
      END

      FUNCTION MixtPerf_Vm_C2Co2G(C2,Co2,G)
      IMPLICIT NONE
      REAL  C2,Co2,G
      REAL  MixtPerf_Vm_C2Co2G
      IF ( Co2 .gt. C2 ) THEN  
         MixtPerf_Vm_C2Co2G = SQRT(2.0/(G - 1.0)*(Co2 - C2))
      ELSE 
         MixtPerf_Vm_C2Co2G = 0.0
      END IF ! Co2
      END

! JH060614 stitched bloodily into cmt-nek. Before that,
!******************************************************************************
!
! RCS Revision history:
!
! $Log: MixtPerf.f,v $
! Revision 1.5  2015/07/17 15:58:14  mrugeshs
!  - MS making the code fortran 77 compatible -- not there yet. regression
!       testing going on
!
! Revision 1.4  2015/07/17 15:21:36  jhackl
! JH071715 more fortran 77 goodness
!
! Revision 1.3  2015/03/03 21:09:25  jhackl
! compiles gfortran 4.8.2
!
! Revision 1.2  2015/02/27 22:13:27  mrugeshs
!  - MS022715 - Added function to compute Cp when Cv and R are known
!
! Revision 1.1  2014/06/30 16:42:11  mrugeshs
! - Add CMT souce code
!
! Revision 1.1.1.1  2014/05/05 21:47:47  tmish
! Initial checkin for rocflu macro.
!
! Revision 1.3  2008/12/06 08:43:31  mtcampbe
! Updated license.
!
! Revision 1.2  2008/11/19 22:16:47  mtcampbe
! Added Illinois Open Source License/Copyright
!
! Revision 1.1  2007/04/09 18:48:32  haselbac
! Initial revision after split from RocfloMP
!
! Revision 1.1  2007/04/09 17:59:25  haselbac
! Initial revision after split from RocfloMP
!
! Revision 1.1  2004/12/01 16:48:47  haselbac
! Initial revision after changing case
!
! Revision 1.2  2002/05/28 13:44:44  haselbac
! Added new functions
!
! Revision 1.1  2002/05/04 16:16:52  haselbac
! Initial revision
!
!******************************************************************************
!NTN-----------JWL
      FUNCTION MixtJWL_SO(DE,EN,pres,AA,BB,R1,R2,MW,MA,MB,OM,rho0)
      IMPLICIT NONE
      REAL DE,EN,pres,AA,BB,R1,R2,MW,MA,MB,OM,rho0
      REAL MixtJWL_SO
      MixtJWL_SO =SQRT((AA*(R1*(rho0/DE)/DE-OM/DE-OM/(R1*rho0)-
     >                MW/(R1*rho0/DE))+
     >        MA*pres/(DE*DE)*(1.-OM/(R1*rho0/DE)))*exp(-R1*rho0/DE)+
     >        (BB*(R2*(rho0/DE)/DE-OM/DE-OM/(R2*rho0)-MW/(R2*rho0/DE))+
     >        MB*pres/(DE*DE)*(1.-OM/(R2*rho0/DE)))*exp(-R2*rho0/DE)+
     >        OM*(EN+pres/DE)+MW*DE*EN)
      END

      FUNCTION MixtJWL_PR(DE,EN,AA,BB,R1,R2,OM,rho0)
      IMPLICIT NONE
      REAL DE,EN,AA,BB,R1,R2,OM,rho0
      REAL MixtJWL_PR
      MixtJWL_PR =AA*(1.-(OM*DE)/(R1*rho0))*exp(-R1*rho0/DE)+
     >            BB*(1.-(OM*DE)/(R2*rho0))*exp(-R2*rho0/DE)+OM*DE*EN
      END

      FUNCTION MixtJWL_TE(cv,DE,EN,AA,BB,R1,R2,rho0)
      IMPLICIT NONE
      REAL cv,DE,EN,AA,R1,R2,rho0,BB
      REAL MixtJWL_TE
      MixtJWL_TE = 1./cv*(EN-AA/(R1*rho0)*exp(-R1*rho0/DE)-
     >             BB/(R2*rho0)*exp(-R2*rho0/DE))
      END
!Temperature as a function of pressure and density
!      FUNCTION MixtJWL_TE(cv,DE,pres,AA,BB,R1,R2,rho0,OM)  
!      IMPLICIT NONE
!      REAL cv,DE,pres,AA,BB,R1,R2,rho0,OM  
!      REAL MixtJWL_TE 
!      MixtJWL_TE=1./OM/cv/DE*(pres-AA*exp(-R1*rho0/DE)-
!     >           BB*exp(-R2*rho0/DE))
!      END
      FUNCTION MixtJWL_I_ENE(DE,PRES,AA,BB,R1,R2,rho0,OM)
      IMPLICIT NONE
      REAL DE,PRES,AA,BB,R1,R2,rho0,OM
      REAL  MixtJWL_I_ENE
      MixtJWL_I_ENE =(pres-(AA*(1.-OM/(R1*rho0/DE))
     > *exp(-R1*rho0/DE)+BB*(1.-OM/(R2*rho0/DE))
     > *exp(-R2*rho0/DE)))/(OM*DE)
      END
