# Example script for reading int_fld data
# Kept similar to pymech
import struct
import numpy as np
import os


class point:
  """class defining point variables"""

  def __init__(self, ldim):
    self.pos = np.zeros((ldim))


class pset:
  """class containing data of the point collection"""

  def __init__(self, ldim, npoints):
    self.ldim = ldim
    self.npoints = npoints
    self.pset = [point(ldim) for il in range(npoints)]


def set_pnt_pos(data, il, lpos):
  """set position of the single point"""
  lptn = data.pset[il]
  data_pos = getattr(lptn, 'pos')
  for jl in range(data.ldim):
    data_pos[jl] = lpos[jl]


def write_int_pos(fname, wdsize, emode, data):
  """ write point positions to the file"""
  # open file
  outfile = open(fname, 'wb')

  # word size
  if (wdsize == 4):
    realtype = 'f'
  elif (wdsize == 8):
    realtype = 'd'

  # header
  header = '#iv1 %1i %1i %10i ' % (wdsize, data.ldim, data.npoints)
  header = header.ljust(32)
  outfile.write(header.encode('utf-8'))

  # write tag (to specify endianness)
  # etagb = struct.pack(emode+'f', 6.54321)
  # outfile.write(etagb)
  outfile.write(struct.pack(emode + 'f', 6.54321))

  # write point positions
  for il in range(data.npoints):
    lptn = data.pset[il]
    data_pos = getattr(lptn, 'pos')
    outfile.write(struct.pack(emode + data.ldim * realtype, *data_pos))


def write_channel(path, Ret, yplus, Lx, Lz, Nx, Nz, lx1):
  """
    Function to write the interploation sensing plane
    Ret : Retau 
    yplus: wall unit distance 
    Lx, Lz: Domain size 
    Nx, Nz: Number of elements 
    lx1: Poly order 
    """

  fname = f'{path}/int_pos'

  #if os.path.exists(fname):
  #    print(f"File Exists!:{fname}", flush=True)
  #    return True

  wdsize = 8
  # little endian
  emode = '<'
  # set of points
  ldim = 3
  channelh = 1.0
  npointsx = Nx * lx1
  npointsz = Nz * lx1
  npointsy = 2
  # Domain Size
  xmin, xmax = 0.0, Lx
  zmin, zmax = 0.0, Lz
  ymin, ymax = 0.0, yplus / Ret * channelh

  # create point coordinates
  ptx = np.linspace(xmin, xmax, npointsx)
  ptz = np.linspace(zmin, zmax, npointsz)
  pty = np.linspace(ymin, ymax, npointsy)
  # allocate space
  npoints = npointsx * npointsz * npointsy
  # number of points
  data = pset(ldim, npoints)
  print('Allocated {0} points'.format(npoints))

  # initialise point position buffer
  lpos = np.zeros(data.ldim)

  # assign point structure
  npoints = 0
  for jy in range(npointsy):
    for ix in range(npointsx):
      for iz in range(npointsz):
        lpos[0] = ptx[ix]
        lpos[1] = pty[jy]
        lpos[2] = ptz[iz]
        set_pnt_pos(data, npoints, lpos)
        npoints = npoints + 1

  # write points to the file
  write_int_pos(fname, wdsize, emode, data)
  print('Written {0} points to the file'.format(npoints))
  return True


if __name__ == "__main__":
  # initialise variables
  fname = '02-run/int_pos'
  wdsize = 8
  # little endian
  emode = '<'
  # big endian
  # emode = '<'
  # set of points
  ldim = 3

  Ret = 180
  channelh = 1.0
  npointsx = 4 * 6
  npointsz = 4 * 6
  npointsy = 2

  xmin, xmax = 0.0, 2.67
  zmin, zmax = 0.0, 0.8
  ymin, ymax = 0.0, 15.0 / Ret * channelh

  # create point coordinates
  ptx = np.linspace(xmin, xmax, npointsx)
  ptz = np.linspace(zmin, zmax, npointsz)
  pty = np.linspace(ymin, ymax, npointsy)
  # allocate space
  npoints = npointsx * npointsz * npointsy
  # number of points
  data = pset(ldim, npoints)
  print('Allocated {0} points'.format(npoints))

  # initialise point position buffer
  lpos = np.zeros(data.ldim)

  # assign point structure
  npoints = 0
  for jy in range(npointsy):
    for ix in range(npointsx):
      for iz in range(npointsz):
        lpos[0] = ptx[ix]
        lpos[1] = pty[jy]
        lpos[2] = ptz[iz]
        set_pnt_pos(data, npoints, lpos)
        npoints = npoints + 1

  # write points to the file
  write_int_pos(fname, wdsize, emode, data)
  print('Written {0} points to the file'.format(npoints))
