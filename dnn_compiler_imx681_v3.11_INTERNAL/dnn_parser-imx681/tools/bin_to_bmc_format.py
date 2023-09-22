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
  description='Convert a binary (.bin) file ' +
  'to a format that can be loaded in BMC memory in the RTL simulation environment ' +
  '(one 8-bit hex value per line)',
 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("infile", help="Path to input file (.txt)")
parser.add_argument("outfile", help="Path to output file (.txt)")

def main():
  args = parser.parse_args()

  in_file = open(args.infile, 'r')
  out_file = open(args.outfile, 'w')

  data = np.fromfile(in_file, dtype=np.uint8)

  for x in data:
    out_file.write("%02x\n" % (np.uint8(x)))

if __name__ == "__main__":
    main()
