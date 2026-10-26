#!/usr/bin/python

import sys, os, re

#mergestr = lambda x: reduce((lambda a,b: a+" "+b),x,"")

pathjoin = lambda a,b: os.path.normpath(os.path.join(a,b))
include_re = re.compile("\s*#\s*include\s*\"([^\"]*)\"")
incmatch = lambda x: ( include_re.match(line) for line in open(x) )
incline = lambda x,m: pathjoin(os.path.split(x)[0],m.group(1))
incl = lambda x: [ incline(x,m) for m in incmatch(x) if m!=None ]
includes = {}
def get_include(x):
	if not includes.has_key(x): includes[x] = incl(x)
	return includes[x]

def closure(seq,f):
	v = [], [x for x in seq], set(x for x in seq)
	while len(v[1]): [(v[1].append(y),v[2].add(y)) for y in 
	  f((lambda x: (v[0].append(x),x)[1])(v[1].pop())) if not y in v[2]]
	return v[0]

src_files = sys.argv[1:]
files = closure(src_files, get_include)
deps = dict((x,closure(includes[x],lambda y: includes[y])) for x in src_files)

obj = lambda x: os.path.splitext(x)[0]+".o"

for x in src_files:
	print obj(x)+": "+x+reduce((lambda a,b: a+" "+b),deps[x],"")

print
print "OBJECTS="+reduce((lambda a,b: a+" "+obj(b)),src_files,"")
