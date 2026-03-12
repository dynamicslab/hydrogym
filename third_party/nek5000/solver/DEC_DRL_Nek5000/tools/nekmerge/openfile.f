c-----------------------------------------------------------------------
      subroutine open_file_in(ifile)

      character*80 file,fout,fbout,string
      character*1  file1(80),fout1(80),fbout1(80),string1(80)
      equivalence (file1,file)
      equivalence (fout1,fout)
      equivalence (fbout1,fbout)
      equivalence (string,string1)

      write(6,*) 'Input old (source) file name:'      
      call blank(file,80)
      read(5,80) file
      len = ltrunc(file,80)
   80 format(a80)
   81 format(80a1)

      call chcopy(file1(len+1),'.rea',4)

      open(unit=10, file=file)

      return
      end
c-----------------------------------------------------------------------
      subroutine open_file_out(ifile)

      character*80 file,fout,fbout,string
      character*1  file1(80),fout1(80),fbout1(80),string1(80)
      equivalence (file1,file)
      equivalence (fout1,fout)
      equivalence (fbout1,fbout)
      equivalence (string,string1)

      write(6,*) 'Input new (output) file name:'      
      call blank(fout,80)
      read(5,80) fout
      fbout = fout
      lou = ltrunc(fout,80)

      call chcopy(fout1(lou+1),'.rea',4)
      call chcopy(fbout1(lou+1),'.re2\0',5)

      open(unit=11, file=fout)
      call byte_open(fbout)

      return
      end
c-----------------------------------------------------------------------
