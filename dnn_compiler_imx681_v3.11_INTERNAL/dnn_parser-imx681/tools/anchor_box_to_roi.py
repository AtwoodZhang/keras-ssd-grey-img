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
import math

# Parser for command line arguments
parser = argparse.ArgumentParser(
  description='Convert anchor boxes to equivalent ROIs for ROI_POOL testing',
 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("infile", help="Path to input file (e.g. fd_anchor_boxes.txt)")
parser.add_argument("outfile", help="Path to output file (.txt)")
parser.add_argument("--width", type=int, default=80, 
  help="Image width")
parser.add_argument("--height", type=int, default=60, 
  help="Image height")

def main():
  args = parser.parse_args()

  in_file = open(args.infile, 'r')
  out_file = open(args.outfile, 'w')

  first_data = True
  for line in in_file:
    # Remove whitespace
    line = line.strip()
    # Ignore comment lines
    if line.startswith("#"):
      continue
    # Split by commas, remove empty entry from the end
    vals = line.split(",")
    vals = vals[:-1]
    # Make sure there are 4 values, and convert them all to ints
    if len(vals) != 4:
      raise RuntimeError("Invalid line in anchor box file: " + line)
    # Skip first line of data (scale factors)
    if first_data:
      first_data = False
    else:
      # Convert from s0.7 format to float
      vals = [float(v)*2.0**-7 for v in vals]
      # Convert to image coordinates
      starty = int(math.floor(vals[0] * args.height))
      startx = int(math.floor(vals[1] * args.width))
      endy = min(args.height, int(math.ceil((vals[0] + vals[2]) * args.height)))
      endx = min(args.width, int(math.ceil((vals[1] + vals[3]) * args.width)))
      out_file.write("%02x\n%02x\n%02x\n%02x\n%02x\n" % (
        0, startx, starty, endx, endy))

if __name__ == "__main__":
    main()
