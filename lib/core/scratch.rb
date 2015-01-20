#FunctionalTestOfMixtureOfModels.rb

require 'rubygems'
require 'mathn'
require 'bundler/setup'
require_relative 'Utilities'
require 'distribution'


# base Gaussian model for estimating a single gaussian's parameters
# This base model is only useful as part of a mixture of models, where some
# models parameters are "more adaptable."
# A Gaussian model to incorporate into a "mixture of models" distribution model
# of a neuron's netInputs or outputs across examples
#
# Using the following iterative method for single pass variance estimation:
# FROM  Wikipedia  http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
#def online_variance(data):
#    n = 0
#mean = 0
#M2 = 0
#
#for x in data:
#  n = n + 1
#  delta = x - mean
#  mean = mean + delta/n
#  M2 = M2 + delta*(x - mean)
#
#  if (n < 2):
#      return 0
#
#  variance = M2/(n - 1)
#  return variance
#end

class GaussModel
  attr_reader :mean, :std, :prior, :bayesNumerator
  include Distribution
  include Distribution::Shorthand

  def initialize(mean, std, prior, numberOfExamples)
    @numberOfExamples = numberOfExamples
    @mean = mean
    @std = std
    @prior = prior
    @bayesNumerator = nil
    @examplesProbability = nil
    @n = 0.0
  end

  # called at the beginning of each epoch
  def initEpoch
    @n = 0.0
  end

  # called for each example
  # @param [real] inputOrOutputForExample net summed input to neuron; or analog output of neuron
  def calculateBayesNumerator(inputOrOutputForExample)
    @bayesNumerator = 0.0
    stdErr = (inputOrOutputForExample - @mean) / @std
    @bayesNumerator = norm_pdf(stdErr).abs * (@prior / @std) if (stdErr.abs < 15.0)
  end

  # called for each example
  # @param [real] bayesDenominator
  def estimateProbabilityOfExample(bayesDenominator)
    @examplesProbability = @bayesNumerator / bayesDenominator
  end

  # called for each example
  # @param [real] x  aka inputOrOutputForExample; net summed input to neuron; or analog output of neuron
  def prepForRecalculatingModelsParams(inputOrOutputForExample=0.0)
    x = inputOrOutputForExample
    @n += @examplesProbability
  end

  # called at the end of each epoch
  def atEpochsEndCalculateModelParams
    @prior = @n / @numberOfExamples
  end

  def to_s
    "\t\tmean=\t#{mean}\tstd=\t#{std}\tprior=\t#{prior}\n"
  end
end


class GaussModelAdaptable < GaussModel

  def initialize(mean, std, prior, numberOfExamples)
    super
    @newMean = nil
    @m2 = nil
    @delta = nil
    self.initEpoch
  end

  def initEpoch
    @n = 0.0
    @newMean = 0.0
    @m2 = 0.0
  end

  def prepForRecalculatingModelsParams(inputOrOutputForExample)
    x = inputOrOutputForExample
    @n += @examplesProbability
    @delta = (x - @newMean) * @examplesProbability
    @newMean = @newMean + (@delta / @n) if (@n > 1.0e-10)
    deltaPrime = (x - @newMean) * @examplesProbability
    @m2 = @m2 + (@delta * deltaPrime)
  end

  def atEpochsEndCalculateModelParams
    @prior = @n / @numberOfExamples
    @mean = @newMean
    variance = @m2/(@n - 1)
    newStd = Math.sqrt(variance)
    @std = newStd
    #@prior = @n / @numberOfExamples
  end

  def to_s
    "\t\tmean=\t#{mean}\tstd=\t#{std}\tprior=\t#{prior}\n"
  end
end


class ExampleDistributionModel
  attr_reader :models

  def initialize(args)
    @args = args
    @numberOfExamples = args[:numberOfExamples]
    @classesOfModels = [GaussModelAdaptable, GaussModelAdaptable, GaussModel]
    @mean = [1.0, -1.0, 0.0]
    @std = [2.0, 2.0, 4.0] # [0.05, 0.05, 4.0]
    @prior = [0.33, 0.33, 0.34]
    @models = []
  end

  # use this method at beginning of EACH META-EPOCH
  def initMetaEpoch
    numberOfModels = @classesOfModels.length
    numberOfModels.times { |i| @models << @classesOfModels[i].new(@mean[i], @std[i], @prior[i], @numberOfExamples) }
  end

  # use this method at beginning of EACH EPOCH  -- YES
  def initEpoch
    @models.each { |model| model.initEpoch }
  end

  # use this method for EACH EXAMPLE   -- YES
  def useExampleToImproveDistributionModel(netInputOrOutput)
    @models.each { |model| model.calculateBayesNumerator(netInputOrOutput) }
    bayesDenominator = @models.inject(0.0) { |sum, model| sum + model.bayesNumerator }
    @models.each do |model|
      model.estimateProbabilityOfExample(bayesDenominator)
      model.prepForRecalculatingModelsParams(netInputOrOutput)
    end
  end

  # use this method at the END of EACH EPOCH -- YES
  def atEpochsEndCalculateModelParams
    @models.each { |model| model.atEpochsEndCalculateModelParams }
  end

  def calcError(netInputOrOutput)
  end

  def to_s
    models.inject("") { |concat, model| concat + model.to_s }
  end
end

# Using Mixture Model to Estimate the distribution a Neuron's netInputs from all examples in an epoch
# This class is one part of of a multi-part learning strategy...
class EstimateInputDistribution < LearningStrategyBase

  def initialize(theEnclosingNeuron, ** strategyArgs)
    super
    classOfInputDistributionModel = @strategyArgs[:classOfInputDistributionModel]
    neuron.inputDistributionModel = classOfInputDistributionModel.new(@strategyArgs) # keeping inputDistributionModel
    # in neuron for continuity across the invoking of different learning strategies.
  end

  def startStrategy
    inputDistributionModel.initMetaEpoch
  end

  def startEpoch
    inputDistributionModel.initEpoch
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    self.netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
    inputDistributionModel.useExampleToImproveDistributionModel(netInput)
  end

  def endEpoch
    inputDistributionModel.atEpochsEndCalculateModelParams
  end

end




# MAIN TESTING...

include Distribution
include Distribution::Shorthand


def fillDataAry(args)
  positiveDist = norm_rng(mean = 0.8, sigma = 0.05)
  negativeDist = norm_rng(mean = -1.3, sigma = 0.05)

  syntheticData = []
  numExamplesFromPositiveDistribution = (0.666 * args[:numberOfExamples]).round
  numExamplesFromNegativeDistribution = args[:numberOfExamples] - numExamplesFromPositiveDistribution

  numExamplesFromPositiveDistribution.times { syntheticData << positiveDist.call }
  numExamplesFromNegativeDistribution.times { syntheticData << negativeDist.call }
  syntheticData.shuffle
end

def doAnEpoch(distributionModel, syntheticData)
  distributionModel.initEpoch
  syntheticData.each do |exampleValue|
    distributionModel.useExampleToImproveDistributionModel(exampleValue)
  end
  distributionModel.atEpochsEndCalculateModelParams
end


## Just creating synthetic data
args = {:numberOfExamples => 300}

syntheticData = fillDataAry(args)

distributionModel = ExampleDistributionModel.new(args)
distributionModel.initMetaEpoch # meta-epoch level


30.times do |i|
  puts distributionModel.to_s
  puts
  doAnEpoch(distributionModel, syntheticData)
end

puts "The Last 3 Lines of the Output (immediately above) Should Be:\n"


puts "mean=	0.7997025687556193	std=	0.0498925461200198	prior=	0.6666666666666666"
puts "mean=	-1.287091313436506	std=	0.05155588532637742	prior=	0.3333333333333333"
puts "mean=	0.0	std=	4.0	prior=	6.384569724959703e-33"
