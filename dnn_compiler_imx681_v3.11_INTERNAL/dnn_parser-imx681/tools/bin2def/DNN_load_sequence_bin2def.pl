#! /usr/bin/perl
use File::Basename;

## Usage ######################################################################
# perl DNN_load_sequence_bin2def.pl <filename.bin>
###############################################################################

print "bin-to-def file format conversion start ...\n";

### open input file ####
my $fname = $ARGV[0];
open(IN1, "<$fname")   || die "cannot open $fname : $! \n";
binmode IN1;
#print "INPUT: $ARGV[0] open => SUCCESS!\n";

### open output file ####
$output_filename = basename($fname);
$output_filename =~ s/.bin/.def/;
open(OUT, ">$output_filename") || die "cannot open $output_filename : $! \n";
#print "OUTPUT: $output_filename open => SUCCESS!\n";


### def file generation ####
printf(OUT "HiAddress       01\n");
my $BLOCK_SIZE=3;
my $i = 0;
while (read(IN1, my $buf, $BLOCK_SIZE, i*$BLOCK_SIZE)){
   my @data = split(//,$buf);
   my $addr  = ord($data[0])*256 + ord($data[1]);
   my $value = ord($data[2]);
   
   printf(OUT "0x%04X    %02Xh    20%04X 0 8\n",$addr,$value,$addr);
   $i++;
}

### closing ###
close(IN1);
close(OUT);
print "... $output_filename genereted [Completed] .\n";

