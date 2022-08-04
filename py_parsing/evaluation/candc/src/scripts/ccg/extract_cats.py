# C&C NLP tools
# Copyright (c) Universities of Edinburgh, Oxford and Sydney
# Copyright (c) James R. Curran
#
# This software is covered by a non-commercial use licence.
# See LICENCE.txt for the full text of the licence.
#
# If LICENCE.txt is not included in this distribution
# please email candc@it.usyd.edu.au to obtain a copy.

import sys

SEP = '|'

def usage(s):
  print >> sys.stderr, s
  print >> sys.stderr, "usage: extract_cats <stagged_file>"
  sys.exit(1)

if len(sys.argv) != 2:
  usage("incorrect number of arguments")

print "# this file was generated by the following command(s):"

cats = {}
for line in open(sys.argv[1]):
 if line.startswith('#'):
   if line.startswith('# this file was generated'):
     continue
   print line,
   continue

 for word in line.split():
   (word, pos, cat) = word.split(SEP)
   cats[cat] = cats.get(cat, 0) + 1

print "# %s" % (' '.join(sys.argv))
print

items = map(lambda x: (x[1], x[0]), cats.items())
items.sort(lambda x, y: cmp(y, x))

for item in items:
  print '%d %s' % item
