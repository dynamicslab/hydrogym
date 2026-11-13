#!/usr/bin/python

import sys, os, re

obj_files = sys.argv[1:]

defined = dict((x,set([])) for x in obj_files)
undefined = dict((x,set([])) for x in obj_files)
nm_re = re.compile("[0-9a-fA-F]*\s*([BCDRTU])\s+([A-Za-z_][A-Za-z_0-9]*)\s*")
def nm_match(x): return ( nm_re.match(line) for line in os.popen('nm -g '+x) )
def nm_line(x,m):
	if m.group(1)=='U': undefined[x].add(m.group(2))
	else: defined[x].add(m.group(2))
[ [ nm_line(x,m) for m in nm_match(x) if m!=None ] for x in obj_files ]

def closure(seq,f):
	v = [], [x for x in seq], set(x for x in seq)
	while len(v[1]): [(v[1].append(y),v[2].add(y)) for y in 
	  f((lambda x: (v[0].append(x),x)[1])(v[1].pop())) if not y in v[2]]
	return v[0]

needs={}
def get_needs(x):
	if not needs.has_key(x):
		needs[x]=[y for y in obj_files if len(defined[y]&undefined[x])]
	return needs[x]
deps = dict((x,closure(get_needs(x),get_needs)) for x in obj_files)

for x in deps:
	print x,'depends on',reduce((lambda a,b: a+" "+b),deps[x],"")
print

results = [ os.path.splitext(x)[0] for x in obj_files if 'main' in defined[x] ]
print "RESULTS="+reduce((lambda a,b: a+" "+b),results,"")
print

def need_X(objs):
	for x in objs:
		if "XOpenDisplay" in undefined[x]: return True
	return False

for x in results:
	objs = deps[x+'.o'];
	if not (x+'.o') in objs: objs.append(x+'.o')
	sobjs = reduce((lambda a,b: a+" "+b),objs,"")
	if need_X(objs):
		print x+":"+sobjs+" ; @echo LINK $@; $(LINKCMD) $^ -lX11 -o $@"
	else:
		print x+":"+sobjs+" ; @echo LINK $@; $(LINKCMD) $^ -o $@"

