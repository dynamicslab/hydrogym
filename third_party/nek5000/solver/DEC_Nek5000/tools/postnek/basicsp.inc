c     Beginning of basicsp.inc**********
c
c     MAXIMUM # OF History values (per point) and max particles
      PARAMETER (MAXS=11,MAXPAR=10,MAXPTS= 2 000 000)
c     PARAMETER (lvrt=maxpts)
      PARAMETER (lvrt=1)
c
c     Maximum resolution for contour plots
c
      PARAMETER (NXXM=40)
      COMMON /FIELD2/ WKV1(MAXPTS),WKV2(MAXPTS),WKV3(MAXPTS)
     $               ,WORK(MAXPTS),WRK2(MAXPTS)
     $               ,vortex(MAXPTS)
     $               ,vrt1(lvrt),vrt2(lvrt),vrt3(lvrt)
      COMMON /FIELD3/ RXM1(MAXPTS),RYM1(MAXPTS),RZM1(MAXPTS)
     $               ,SXM1(MAXPTS),SYM1(MAXPTS),SZM1(MAXPTS)
     $               ,TXM1(MAXPTS),TYM1(MAXPTS),TZM1(MAXPTS)
     $               ,JACM1(MAXPTS)
 
      COMMON /FIELDr/ mult(MAXPTS)
      real mult
 
      parameter (msrf1=nxm*nzm*6*nelm)
      parameter (msrf=min(msrf1,maxpts))
      common /fields/ area(msrf),unx(msrf),uny(msrf),unz(msrf)
 
      COMMON /FIELDi/ nglob,iglob(maxpts),nsrf,srf_list(msrf)
      integer nsrf,srf_list
 
      PARAMETER (mpscal=4) ! Max number of passive scalars
      COMMON /FIELD1/ SDUMP(MAXPTS,8+mpscal)
      REAL U (MAXPTS),V (MAXPTS),W (MAXPTS)
     $    ,P (MAXPTS),T (MAXPTS)
     $    ,XP(MAXPTS),YP(MAXPTS),ZP(MAXPTS)
     $    ,PSA(MAXPTS)
     $    ,PSB(MAXPTS)
     $    ,PSC(MAXPTS)
     $    ,PSD(MAXPTS)
      EQUIVALENCE  (XP,SDUMP(1,1)) , (YP,SDUMP(1,2)) , (ZP,SDUMP(1,3))
     $            ,(U ,SDUMP(1,4)) , (V ,SDUMP(1,5)) , (W ,SDUMP(1,6))
     $            ,(P ,SDUMP(1,7)) , (T ,SDUMP(1,8))
     $            ,(PSA,SDUMP(1, 9))
     $            ,(PSB,SDUMP(1,10))
     $            ,(PSC,SDUMP(1,11))
     $            ,(PSD,SDUMP(1,12))
 
      COMMON /OLDSTF/ TIMEH(MAXS),UH(MAXS,7),VH(MAXS,7),TH(MAXS,7)
     $               ,WH(MAXS,7),PSH(MAXS,7)
     $               ,PH(MAXS,7),COURNT(MAXS),FLOW(MAXS),PGRAD(MAXS)
     $               ,TIMEX(MAXS)
      REAL JACM1
 
      PARAMETER (NPRT=100)
      PARAMETER (NSTOR=400*NPRT)
c     PARAMETER (NPRT=10)
c     PARAMETER (NSTOR=40*NPRT)
      PARAMETER (NLNMAX=280000)
      COMMON /RLINES/ XLINES(2,NLNMAX),YLINES(2,NLNMAX),ZLINES(2,NLNMAX)
      COMMON /ILINES/ NLINES
      COMMON /PARTCL/ XSTR(5,NSTOR),DPT(NPRT),XYZ0(5,NPRT)
      COMMON /IPRTCL/ IOSTP(NPRT),IPOFF(NPRT)
     $               ,NSTRP(NPRT),NSTR0(NPRT),ICST(NPRT)
      COMMON /RBUBBL/ TBURST(2,NPRT),TCLOCK(3),SCLOCK
      COMMON /IBUBBL/ NBURST,NCLOCK,NPART
      COMMON /LGRNG/ XBMIN(NELM),XBMAX(NELM)
     $              ,YBMIN(NELM),YBMAX(NELM),ZBMIN(NELM),ZBMAX(NELM)
     $              ,RR(3,NPRT),XLG(4,NPRT),IELGR(NPRT),ILOST(NPRT)
     $              ,NLP,TOLLG,SCALLG(5,NPRT)
      COMMON /LGRLOG/ IFOUTL,IFSCND,IFSELE,IFUSCL,IFTHRS,IFVLPR,ifsubset
      LOGICAL         IFOUTL,IFSCND,IFSELE,IFUSCL,IFTHRS,IFVLPR,ifsubset
      COMMON /PLTLOG/ ifportrait,ifwkclobber
      LOGICAL         ifportrait,ifwkclobber
      COMMON /LGRINT/ NPLEL,LEVMIN,LEVMAX
      COMMON /LGRREA/ USRMIN,USRMAX,THRMIN,THRMAX
 
      parameter (max_cont_nlev=1000)
      COMMON /contcmr/ spec_cont(0:max_cont_nlev)
     $               , spec_ind (0:max_cont_nlev) 
      COMMON /contcmi/ spec_cont_nlev,jhard
      COMMON /contcml/ if_spec_cont
      logical if_spec_cont
      integer spec_cont_nlev,jhard
c
c     Common for Identifying/Tracking Vortices
c
      INTEGER NVEMX
      PARAMETER(NVEMX=NELM)
      integer nfacemx
      PARAMETER(nfacemx=300)
C     nfacemx ~>=  (nelm)**(2/3)
      integer nhemimx
      PARAMETER(nhemimx=30)
 
      INTEGER nhemi
      INTEGER bttm_elist(nfacemx,2)
      parameter (mxel=400 000)
      INTEGER  IXV(NVEMX),IYV(NVEMX),IZV(NVEMX),IEV(NELM),isubset(mxel)
      INTEGER  CCOLOR(NELM),JCC(NELM+1),CORE_COLOR(NELM)
      INTEGER  ieminl2(NELM)
c
c     Object lists (for hemispher surfaces)
c
      COMMON /VRTXTRKi/ NVE,IXV,IYV,IZV,IEV
     $                 ,ccolor,jcc,core_color
     $                 ,nccolor,ieminl2,nhemi,bttm_elist
     $                 ,list_obi(msobjs+1)
     $                 ,list_obj(msobjs*6*nelm)
 
      REAL     LAM2(NELM)
      REAL XPC(NELM),YPC(NELM),ZPC(NELM)
      REAL ENSTROPHY(NELM)
      REAL strss1(nfacemx),strss2(nfacemx),strss3(nfacemx)

 
      COMMON /VRTXTRKr/ LAM2,XPC,YPC,ZPC,ENSTROPHY
 
      parameter (mxpro=4000)  ! Maximum number of profile pts
      common /ezdat/ xpt(0:mxpro),ypt(0:mxpro),zpt(0:mxpro)
     $             , upr(0:mxpro),vpr(0:mxpro),wpr(0:mxpro)
     $             , ppr(0:mxpro),tep(0:mxpro),vtx(0:mxpro)
     $             , wkp(0:mxpro)
     $             , v1p(0:mxpro),v2p(0:mxpro),v3p(0:mxpro)
     $             , dpf(0:mxpro),pro(0:mxpro)
     $             , outprof(15)
      real prodat(0:mxpro,15)
      equivalence 
     $         (prodat(0, 1),xpt),(prodat(0, 2),ypt),(prodat(0, 3),zpt)
     $       , (prodat(0, 4),upr),(prodat(0, 5),vpr),(prodat(0, 6),wpr)
     $       , (prodat(0, 7),ppr),(prodat(0, 8),tep),(prodat(0, 9),vtx)
     $       , (prodat(0,10),wkp)
     $       , (prodat(0,11),v1p),(prodat(0,12),v2p),(prodat(0,13),v3p)
     $       , (prodat(0,14),dpf)
     $       , (prodat(0,15),pro)
      common /ezdati/ itst(0:mxpro),nprof_outs,indx_prof(15)
      common /ezdatl/ ifxprf,ifyprf,ifzprf,ifuprf,ifvprf,ifwprf
     $               ,ifpprf,iftprf,ifvtpf,ifwkpf,ifv1pf,ifv2pf,ifv3pf
     $               ,ifdprf,ifmtpf
     $               ,if_auto_box
      logical         ifxprf,ifyprf,ifzprf,ifuprf,ifvprf,ifwprf
     $               ,ifpprf,iftprf,ifvtpf,ifwkpf,ifv1pf,ifv2pf,ifv3pf
     $               ,ifdprf,ifmtpf
     $               ,if_auto_box
      logical         ifprofs(15)
      equivalence    (ifxprf,ifprofs)! Q:? Will this equiv. ifyprf, etc. ?
c
      common/imgc_log/if_img_imcopy,if_img_convert,if_img_tif,if_img_rgb
      logical         if_img_imcopy,if_img_convert,if_img_tif,if_img_rgb
c
c     End of basicsp.inc**********
c-----------------------------------------------------------------------
