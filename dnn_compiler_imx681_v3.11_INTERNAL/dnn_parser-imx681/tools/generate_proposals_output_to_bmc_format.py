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
  description='Convert expected output of generate proposals (from C model or ' +
  'PyTorch) to the format expected by Firmware',
 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("infile", help="Path to input file (.txt)")
parser.add_argument("outfile", help="Path to output file (.txt)")
parser.add_argument("--offset", type=int, default=0, 
  help="Starting offset of data to extract")
parser.add_argument("--rowstride", type=int, default=0, 
  help="Row stride in bytes (0 for full file)")
parser.add_argument("--coord_to_size", action="store_true",
  help="If true, convert end coordinates (x2, y2) to size (width, height)")
parser.add_argument("--size_to_coord", action="store_true",
  help="If true, convert size (width, height) to end coordinates (x2, y2)")
def main():
  args = parser.parse_args()

  in_file = open(args.infile, 'r')
  out_file = open(args.outfile, 'w')

  lines = in_file.readlines()
  rowsize = 4
  if args.rowstride == 0:
    args.rowstride = rowsize

  num_rois = int(len(lines)/args.rowstride)
  out_file.write("%02x\n" % (np.uint8(0)))
  out_file.write("%02x\n" % (np.uint8(0)))
  out_file.write("%02x\n" % (np.uint8(0)))
  out_file.write("%02x\n" % (np.uint8(num_rois)))
  for i in range(0, num_rois):
    x = int(lines[args.offset + i*args.rowstride])
    y = int(lines[args.offset + i*args.rowstride + 1])
    w = int(lines[args.offset + i*args.rowstride + 2])
    h = int(lines[args.offset + i*args.rowstride + 3])
    if args.coord_to_size:
      w = w - x + 1
      h = h - y + 1
    if args.size_to_coord:
      w = x + w
      h = y + h
    out_file.write("%02x\n" % (np.uint8(x - 128)))
    out_file.write("%02x\n" % (np.uint8(y - 128)))
    out_file.write("%02x\n" % (np.uint8(w - 128)))
    out_file.write("%02x\n" % (np.uint8(h - 128)))

if __name__ == "__main__":
    main()
