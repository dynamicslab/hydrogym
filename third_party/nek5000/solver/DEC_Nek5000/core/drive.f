      program NEKTON
c
c NEK5000: Spectral Element Computational Fluid Dynamics Solver
c COPYRIGHT (c) 2008-2010 UCHICAGO ARGONNE, LLC
c
c The UChicago Argonne, LLC as Operator of Argonne National 
c Laboratory holds copyright in the Software. The copyright holder 
c reserves all rights except those expressly granted to licensees,
c and U.S. Government license rights.
c 
c License
c 
c    NEK5000 is free software: you can redistribute it and/or modify
c    it under the terms of the GNU General Public License as published by
c    the Free Software Foundation, either version 3 of the License, or
c    (at your option) any later version.
c
c    NEK5000 is distributed in the hope that it will be useful,
c    but WITHOUT ANY WARRANTY; without even the implied warranty of
c    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
c    GNU General Public License for more details.
c
c    You should have received a copy of the GNU General Public License
c    along with NEK5000.  If not, see <http://www.gnu.org/licenses/>.
c
      call nek_init(intracomm)
      call nek_solve()
      call nek_end()

      call exitt()

      end
