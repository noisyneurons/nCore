#FunctionalTestOfMixtureOfModels.rb

require 'rubygems'
require 'mathn'
require 'bundler/setup'
require_relative 'Utilities'
require 'distribution'


#ary = [1.0,2.0,3.0]
#puts ary.mean
#
#puts ary.standardError

#vUnweightedErrors = Vector.elements(ary,false)
#vWeights = Vector[1.0,2.0,10.0]
#
#vWeightedErrors = vUnweightedErrors.collect2(vWeights) {|e1,e2| e1*e2}
#
#puts "vWeightedErrors=\t#{vWeightedErrors}"



allInputs = [1.0, 4.0, 7.0]
#allProbabilities = [1.0, 1.0, 1.0]
allProbabilities = [2.0, 0.000001, 2.0]

sumOfProbabilities = allProbabilities.sum

weightedSum = allInputs.to_v.inner_product(allProbabilities.to_v)
weightedMean = weightedSum / sumOfProbabilities
puts "weightedMean= #{weightedMean}"


# weightedMean = 2.0


weights = allProbabilities.to_v

unweightedErrors = (allInputs.to_v).collect {|input| input - weightedMean}
weightedErrors = (unweightedErrors.collect2(weights) {|error, weight| error * weight}).to_v
sumSquareWeightedErrors = weightedErrors.inner_product(weightedErrors)
sumSquaredWeights = weights.inner_product(weights)

std = Math.sqrt(sumSquareWeightedErrors / sumSquaredWeights)

puts "std=\t#{std}"

