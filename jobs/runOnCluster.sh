#!/bin/bash
### Quick Verification of transfer!
#$ -o ./output
#$ -e ./error
#$ -t 1-10
ruby1.9.3 ~/Code/Ruby/NN2012/nCore/bin/CircleBPofFlockError.rb

