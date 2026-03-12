# Example script for reading point time statistics data
# Kept similar to pymech
import struct
import numpy as np

class point:
    """class defining point variables"""

    def __init__(self,ldim,ntsnap,nfld):
        self.glid = np.zeros((1), dtype=np.uint32)
        self.pos = np.zeros((ldim))
        self.fld = np.zeros((ntsnap,nfld))

class pset:
    """class containing data of the point collection"""

    def __init__(self,ldim,ntsnap,nfld,npoints):
        self.ldim = ldim
        self.ntsnap = ntsnap
        self.nfld = nfld
        self.npoints = npoints
        self.time = []
        self.tmlist = np.zeros((ntsnap))
        self.pset = [point(ldim,ntsnap,nfld) for il in range(npoints)]

def read_int(infile,emode,nvar):
    """read integer array"""
    isize = 4
    llist = infile.read(isize*nvar)
    llist = list(struct.unpack(emode+nvar*'i', llist))
    return llist

def read_flt(infile,emode,wdsize,nvar):
    """read real array"""
    if (wdsize == 4):
        realtype = 'f'
    elif (wdsize == 8):
        realtype = 'd'
    llist = infile.read(wdsize*nvar)
    llist = np.frombuffer(llist, dtype=emode+realtype, count=nvar)
    return llist

def read_int_fld(fname):
    """read data from interpolation file"""
    # open file
    infile = open(fname, 'rb')
    # read header
    header = infile.read(132).split()

    # extract word size
    wdsize = int(header[1])

    # identify endian encoding
    etagb = infile.read(4)
    etagL = struct.unpack('<f', etagb)[0]; etagL = int(etagL*1e5)/1e5
    etagB = struct.unpack('>f', etagb)[0]; etagB = int(etagB*1e5)/1e5
    if (etagL == 6.54321):
        emode = '<'
    elif (etagB == 6.54321):
        emode = '>'

    # get simulation parameters
    ldim = int(header[2])
    npoints = int(header[4])
    ntsnap = int(header[5])
    nfld = int(header[6])
    time = float(header[7])

    # create main data structure
    data = pset(ldim,ntsnap,nfld,npoints)

    # fill simulation parameters
    data.time = time

    # read snapshot time list
    data.tmlist = read_flt(infile,emode,wdsize,data.ntsnap)

    # read global point number
    glidlist = read_int(infile,emode,data.npoints)
    # fill data structure in
    for il in range(data.npoints):
        lptn = data.pset[il]
        data_glid = getattr(lptn,'glid')
        data_glid[0] = glidlist[il]

    # read coordinates
    for il in range(data.npoints):
        lpos = read_flt(infile,emode,wdsize,data.ldim)
        lptn = data.pset[il]
        data_pos = getattr(lptn,'pos')
        for jl in range(data.ldim):
            data_pos[jl] =  lpos[jl]

    # read fields
    for il in range(data.npoints):
        lptn = data.pset[il]
        data_fld = getattr(lptn,'fld')
        for jl in range(data.ntsnap):
            lfld = read_flt(infile,emode,wdsize,data.nfld)
            for kl in range(data.nfld):
                data_fld[jl][kl] = lfld[kl]

    # place for sorting

    return data

def print_sim_data(data):
    """print simulation data"""
    print('Simulation data:')
    print('    ldim = {0}'.format(data.ldim))
    print('    ntsnap = {0}'.format(data.ntsnap))
    print('    nfld = {0}'.format(data.nfld))
    print('    npoints = {0}'.format(data.npoints))
    print('    time = {0}'.format(data.time))
    print('Time snapshots:')
    for il in range(data.ntsnap):
        print('    time{0} = {1}'.format(il+1,data.tmlist[il]))

def print_point_data(data,il):
    """print data related to a single point"""
    print('Point data, npt = {0}'.format(il+1))
    lptn = data.pset[il]
    data_glid = getattr(lptn,'glid')
    print(data_glid)
    data_pos = getattr(lptn,'pos')
    print(data_pos)
    data_fld = getattr(lptn,'fld')
    print(data_fld)

if __name__ == "__main__":
    # in genera there should be loop over files and some concatenation mechanism
    fname = 'ptsphill0.f00002'
    data = read_int_fld(fname)

    print_sim_data(data)

    #for testing
    for il in range(5):
        print_point_data(data,il)

    # place for operations
