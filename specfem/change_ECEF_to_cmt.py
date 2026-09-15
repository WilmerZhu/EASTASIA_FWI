#!/usr/bin/env python

import sys
import numpy as np
import pyproj

ecef_file = str(sys.argv[1])
out_file = str(sys.argv[2])

with open(ecef_file, 'r') as f:
  lines = [ x for x in f.readlines() if not(x.startswith('#')) ]

header = lines[0].split()

lines = [x.split(":") for x in lines]
event_id = lines[1][1].strip()

# initialize pyproj objects
geod = pyproj.Geod(ellps='WGS84')
ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

tau = float(lines[3][1])*1.628
x = float(lines[4][1])
y = float(lines[5][1])
z = float(lines[6][1])

# convert from ECEF to lla
lon, lat, alt = pyproj.transform(ecef, lla, x, y, z)
dep = -alt / 1000.0

# moment tensor
# r, theta, ph- -> x,y,z
# harvard cmt use dyn*cm for moment tensor, *1e7 from N*m
mxx = float(lines[7][1])
myy = float(lines[8][1])
mzz = float(lines[9][1])
mxy = float(lines[10][1])
mxz = float(lines[11][1])
myz = float(lines[12][1])
mt_xyz = np.array([[mxx, mxy, mxz], [mxy, myy, myz], [mxz, myz, mzz]]) * 1.0e7

r = (x**2 + y**2 + z**2)**0.5
theta = np.arccos(z/r)
phi = np.arctan2(y, x)

sthe = np.sin(theta)
cthe = np.cos(theta)
sphi = np.sin(phi)
cphi = np.cos(phi)

a = np.array(
    [ [ sthe*cphi, cthe*cphi, -1.0*sphi ],
      [ sthe*sphi, cthe*sphi,      cphi ],
      [ cthe     , -1.0*sthe,      0.0  ] ])

mt_rtp = np.dot(np.dot(np.transpose(a), mt_xyz), a)

#write out new CMTSOLUTION_cmt
with open(out_file, 'w') as fp:
  fp.write('%s\n'            % ' '.join(header))
  fp.write('%-18s %s\n'      % ('event name:',event_id))
  fp.write('%-18s %+15.8E\n' % ('time shift:',    0.0))
  fp.write('%-18s %+15.8E\n' % ('half duration:',   tau))
  fp.write('%-18s %+15.8E\n' % ('latitude:',    lat))
  fp.write('%-18s %+15.8E\n' % ('longitude:',   lon))
  fp.write('%-18s %+15.8E\n' % ('depth:',      dep))
  fp.write('%-18s %+15.8E\n' % ('Mrr:', mt_rtp[0,0]))
  fp.write('%-18s %+15.8E\n' % ('Mtt:', mt_rtp[1,1]))
  fp.write('%-18s %+15.8E\n' % ('Mpp:', mt_rtp[2,2]))
  fp.write('%-18s %+15.8E\n' % ('Mrt:', mt_rtp[0,1]))
  fp.write('%-18s %+15.8E\n' % ('Mrp:', mt_rtp[0,2]))
  fp.write('%-18s %+15.8E\n' % ('Mtp:', mt_rtp[1,2]))
