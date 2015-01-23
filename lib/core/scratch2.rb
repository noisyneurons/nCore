#FunctionalTestOfMixtureOfModels.rb

require 'rubygems'
require 'mathn'
require 'bundler/setup'
require_relative 'Utilities'
require_relative 'DataSet'
#require 'distribution'


logger = StringIO.new

args = {
    :experimentNumber => $globalExperimentNumber,
    :descriptionOfExperiment => "Proj5MixtureModel; single neuron test of mixture model and related learning strategy components",
    :baseRandomNumberSeed => 0,

    # Training Set parameters
    :numberOfExamples => 16,
    :numberOfTestingExamples => 160,
    :standardDeviationOfAddedGaussianNoise => 0.0,
    :verticalShift => 0.0,
    :horizontalShift => 0.0,
    :angleOfClockwiseRotationOfInputData => 0.0,

    # Results and debugging information storage/access
    :logger => logger
}

dsGen = Generate4ClassDataSet.new(args)

examples = dsGen.generate(numberOfExamples=16, standardDeviationOfAddedGaussianNoise= 0.0)

puts examples

