#! /usr/bin/perl
use File::Basename;

## Usage ######################################################################
# perl DNN_memory_bin2def.pl <filename.bin> <start address>
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


### address ####
my $start_address = hex($ARGV[1]);
my $address_msb = $start_address >> 12;
my $address_lsb = $start_address - ($address_msb<<12);


### def file generation ####
printf(OUT "HiAddress       01\n");

my $grmpa = $address_msb;
printf(OUT "0x3E28    %02Xh    203E28 0 8\n",($grmpa>>12) % 256);
printf(OUT "0x3E29    %02Xh    203E29 0 8\n",($grmpa>>4)  % 256);
printf(OUT "0x3E2A    %02Xh    203E2A 0 8\n",($grmpa%16) << 4  );

my $BLOCK_SIZE=1;
my $addr = $address_lsb;
my $i = 0;
while (read(IN1, my $buf, $BLOCK_SIZE, i*$BLOCK_SIZE)){
   my @data = split(//,$buf);
   my $value = ord($data[0]);
   
   if($addr == 4096){
      $addr = 0;
      $grmpa++;
      printf(OUT "0x3E28    %02Xh    203E28 0 8\n",($grmpa>>12) % 256);
      printf(OUT "0x3E29    %02Xh    203E29 0 8\n",($grmpa>>4)  % 256);
      printf(OUT "0x3E2A    %02Xh    203E2A 0 8\n",($grmpa%16) << 4  );
   }
   printf(OUT "0xA%03X    %02Xh    20A%03X 0 8\n",$addr,$value,$addr);
   $addr++;
   $i++;
}

### closing ###
close(IN1);
close(OUT);
print "... $output_filename genereted [Completed] .\n";

