#FunctionalTestOfMixtureOfModels.rb

require 'rubygems'
require 'mathn'
require 'bundler/setup'
require_relative 'Utilities'
require 'distribution'


ary = [1.0,2.0,3.0]
#puts ary.mean
#
#puts ary.standardError

vUnweightedErrors = Vector.elements(ary,false)
vWeights = Vector[1.0,2.0,10.0]

vWeightedErrors = vUnweightedErrors.collect2(vWeights) {|e1,e2| e1*e2}

puts "vWeightedErrors=\t#{vWeightedErrors}"
