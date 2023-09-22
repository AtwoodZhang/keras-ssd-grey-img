# ------------------------------------------------------------------------------
# Copyright 2020 Sony Semiconductor Solutions Corporation.
# This is UNPUBLISHED PROPRIETARY SOURCE CODE of
# Sony Semiconductor Solutions Corporation.
# No part of this file may be copied, modified, sold, and distributed in any
# form or by any means without prior explicit permission in writing of
# Sony Semiconductor Solutions Corporation.
# ------------------------------------------------------------------------------
import glob
import os
import shutil
import sys

from git import Repo
from subprocess import Popen
from zipfile import ZipFile

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

# Name of logfile created from pyarmor
PYARMOR_LOG_FILENAME = "pyarmor.log"

# Command used to create obfuscated source code
PYARMOR_REGISTER_COMMAND = ["pyarmor", "register", "pyarmor-regcode-2906.txt"]
OBFUSCATE_COMMAND = ["pyarmor", "obfuscate", "--no-cross-protection", "--restrict=0", 
                     "--platform=windows.x86_64,linux.x86_64", "--recursive", 
                     "--exclude", "tools", "--exclude", "build_release.py", 
                     "--exclude", "sample_code",
                     "dnn_compiler.py"]

# Path to file containing constants that will be modified as part of building
# the release
CONSTANTS_FILE = "internal/constants.py"

# Names of variables in CONSTANTS_FILE that may be modified as part of the build 
# process
VERSION_VARIABLE_NAME = "DNN_COMPILER_VERSION"
INTERNAL_USE_VARIABLE_NAME = "DNN_COMPILER_INTERNAL_USE"

# Readme to include in the release
README_FILE = "doc/RELEASE_README.md"

# Other non-source files to include in release package
NONSOURCE_FILES = [
  "configs/base/dnn/pytorch/sony_classification.inc",
  "configs/base/dnn/tflite/sony_detection.inc",
  "configs/base/dnn/tflite/sony_human_existence.inc",
  "configs/base/sensor/imx681.inc",
  "configs/imx681_pytorch_classification_i2c.cfg",
  "configs/imx681_pytorch_classification_fx_i2c.cfg",
  "configs/imx681_tflite_detection_i2c.cfg",
  "configs/imx681_tflite_human_existence_i2c.cfg",
  "data/sony_detection_anchor_boxes.txt",
  "models/sony_detection.tflite",
  "sample_code/pytorch_classification/*",
  "sample_code/pytorch_classification_fx/*",
  "tools/bin2def/*",
]

# Prefix to use for the output release archive. It will be named:
#  [RELEASE_ARCHIVE_PREFIX][version-name].zip
RELEASE_ARCHIVE_PREFIX = "dnn_compiler_v"

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def ReadConstantsFromFile(filename):
  """
  Read the constants file and extract relevant information

  Returns:
  version (float), internal_use (bool), lines (list), version_line (int), internal_use_line (int)
  """
  try:
    fh = open(CONSTANTS_FILE, "r")
  except OSError as e:
    print("ERROR: Failed to open constants source file: %s. %s" % (CONSTANTS_FILE, e))
    sys.exit(-1)

  lines = fh.readlines()
  version_line = -1
  internal_use_line = -1
  for i in range(0, len(lines)):
    tokens = lines[i].split("=")
    if tokens[0].strip() == VERSION_VARIABLE_NAME:
      version = float(tokens[1].strip().strip("\""))
      version_line = i
    elif tokens[0].strip() == INTERNAL_USE_VARIABLE_NAME:
      internal_use = (tokens[1].strip() == "True")
      internal_use_line = i
    if version_line >= 0 and internal_use_line >= 0:
      break
  if version_line < 0:
    print("ERROR: Could not find variable %s in %s" % (VERSION_VARIABLE_NAME, CONSTANTS_FILE))
    sys.exit(-1)
  elif internal_use_line < 0:
    print("ERROR: Could not find variable %s in %s" % (INTERNAL_USE_VARIABLE_NAME, CONSTANTS_FILE))
    sys.exit(-1)
  return version, internal_use, lines, version_line, internal_use_line

def GetUserInput(prev_version):
  """
  Prompt user for inputs

  Returns:
  version (float), do_git (bool)
  """
  default_version = prev_version + 0.01
  while True:
    version = input("Enter version number [prev:%.2f, default:%.2f]:" % (prev_version, default_version))
    if not version:
      version = default_version
    try:
      version = float(version)
      break
    except ValueError:
      print("Enter a valid floating point number X.YY")

  while True:
    do_git = input("Automatically checkin and create tag in git? [y/n]:")
    if do_git in ["y", "Y"]:
      do_git = True
      break
    elif do_git in ["n", "N"]:
      do_git = False
      break
    else:
      print("Enter one of the following: [y/n]")

  return version, do_git    

def WriteReleaseArchive(out_filename, readme_filename, nonsource_files):
  """
  Create a release archive
  """
  with ZipFile(out_filename, 'w') as zip_obj:
    # Add all obfuscated source files from the "dist" directory (created by PyArmor)
    for dir_name, sub_dir_name, filenames in os.walk("dist"):
      for filename in filenames:
        # create complete filepath of file in directory
        path = os.path.join(dir_name, filename)
        relpath = os.path.relpath(path, "dist")
        # Add file to zip
        zip_obj.write(path, relpath)

    # Add readme
    zip_obj.write(README_FILE, "README.md")

    # Add all non-source files, preserving directory structure and allowing wildcards
    for f in NONSOURCE_FILES:
      for file in glob.glob(f):
        relative_path = os.path.split(file)[0]
        zip_obj.write(file)

# ------------------------------------------------------------------------------
# Start of script
# ------------------------------------------------------------------------------
# Check if the repository is clean
repo = Repo.init(".")
if repo.is_dirty():
  proceed = input("WARNING: Repository has modified files. Proceed? [y/n]")
  if not proceed in ["y", "Y"]:
    sys.exit(-1)


# Extract constants from source code
prev_version, internal_use, lines, vline, iline = ReadConstantsFromFile(CONSTANTS_FILE)

# Prompt user for input
version, do_git = GetUserInput(prev_version)

# Update constants in source code
lines[vline] = "%s = %.2f\n" % (VERSION_VARIABLE_NAME, version)
if internal_use:
  lines[iline] = "%s = False\n" % (INTERNAL_USE_VARIABLE_NAME)
with open(CONSTANTS_FILE, "w") as fh:
  fh.writelines(lines)

# If dist directory exists, remove it now
if os.path.exists("dist"):
  shutil.rmtree("dist")

# Register PyTorch license
print("Registering PyTorch license...")
log = open(PYARMOR_LOG_FILENAME, "w")
with Popen(PYARMOR_REGISTER_COMMAND, stdout=log, stderr=log) as proc:
  outs, errs = proc.communicate()
if errs:
  for e in errs:
    print(e)
  print("ERROR: PyArmor registration failed! See %s for more details" % (
    PYARMOR_LOG_FILENAME))
  sys.exit(-1)


# Obfuscate source code, which will create
print("Obfuscating source code using PyArmor...")
log = open(PYARMOR_LOG_FILENAME, "w")
with Popen(OBFUSCATE_COMMAND, stdout=log, stderr=log) as proc:
  outs, errs = proc.communicate()
if errs:
  for e in errs:
    print(e)
  print("ERROR: Obfuscating code using PyArmor failed! See %s for more details" % (
    PYARMOR_LOG_FILENAME))
  sys.exit(-1)

# Create archive containing obfuscated source plus additional files
print("Building release archive...")
output_filename = "%s%.2f.zip" % (RELEASE_ARCHIVE_PREFIX, version)
if os.path.exists(output_filename):
  os.remove(output_filename)
WriteReleaseArchive(output_filename, README_FILE, NONSOURCE_FILES)

# Revert internal_use
if internal_use:
  lines[iline] = "%s = True\n" % (INTERNAL_USE_VARIABLE_NAME)
  with open(CONSTANTS_FILE, "w") as fh:
    fh.writelines(lines)

if do_git:
  # If constants changed, commit changes now
  print("Committing to git repository...")
  repo.git.add(CONSTANTS_FILE)
  repo.index.commit("Updated constants for version number: %.2f" % version)
  repo.remote("origin").push()
  # Tag the repository
  tag_name = "RELEASE_V%.2f" % version
  print("Tagging git repository: %s..." % tag_name)
  new_tag = repo.create_tag(tag_name, message="Automatic version tag")
  repo.remote("origin").push(new_tag)

print("Done. Release archive created: %s" % output_filename)
