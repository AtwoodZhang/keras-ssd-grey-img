# ------------------------------------------------------------------------------
# Copyright 2021 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
import argparse
import numpy as np
import os

# Parser for command line arguments
parser = argparse.ArgumentParser(
  description='Convert a file in the C Model\'s ground truth format ' +
  '(one 8-bit decimal value per line) ' +
  'to a format that can be loaded in BMC memory in the RTL simulation environment ' +
  '(one 8-bit hex value per line)',
 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("infile", help="Path to input file (.txt)")
parser.add_argument("outfile", help="Path to output file (.txt)")
parser.add_argument("--offset", type=int, default=0, 
  help="Starting offset of data to extract")
parser.add_argument("--rowsize", type=int, default=0, 
  help="Number of bytes per row (0 for full file)")
parser.add_argument("--rowstride", type=int, default=0, 
  help="Row stride in bytes (0 for full file)")
parser.add_argument("--nrows", type=int, default=1, 
  help="Number of rows to extract")
parser.add_argument("--convert_unsigned", action="store_true",
  help="If true, convert values from signed to unsigned")

def main():
  args = parser.parse_args()

  in_file = open(args.infile, 'r')
  out_file = open(args.outfile, 'w')

  lines = in_file.readlines()
  if args.rowsize == 0:
    args.rowsize = len(lines)
  if args.rowstride == 0:
    args.rowstride = len(lines)

  for i in range(0, args.nrows*args.rowstride):
    if i % args.rowstride < args.rowsize:
      if args.convert_unsigned:
        val = int(lines[args.offset+i])+128
      else:
        val = int(lines[args.offset+i])
      out_file.write("%02x\n" % (np.uint8(val)))

if __name__ == "__main__":
    main()
