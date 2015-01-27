### VERSION "nCore"
; ## ../nCore/lib/core/NeuronLearningStrategies.rb

require 'distribution'

###################################################################
###################################################################
; ####################### Learning Strategies #######################

class LearningStrategyBase # strategy for standard bp learning for output neurons
  attr_reader :neuron, :strategyArgs, :inputLinks, :outputLinks
  include CommonNeuronCalculations

  def initialize(theEnclosingNeuron, ** strategyArgs)
    @neuron = theEnclosingNeuron
    @strategyArgs = strategyArgs
    ioFunction = @strategyArgs[:ioFunction]
    neuron.extend(ioFunction)
    self.extend(ioFunction)
    neuron.output = ioFunction(neuron.netInput) # only needed for recurrent simulators
    @inputLinks = @neuron.inputLinks
    @outputLinks = @neuron.outputLinks if @neuron.respond_to?(:outputLinks)
  end

  def startStrategy
  end

  def startEpoch
  end

  def propagate(exampleNumber)
  end

  def learnExample
  end

  def endEpoch
  end

  # service routines that may be used by various learning strategies
  def calcWeightsForUNNormalizedInputs
    biasWeight = inputLinks.inject(0.0) { |sum, link| sum + link.propagateUsingZeroInput }
    inputLinks.each { |aLink| aLink.calcWeightsForUNNormalizedInputs }
    inputLinks[-1].weight = biasWeight
    inputLinks.each { |aLink| aLink.resetAllNormalizationVariables }
  end


  # simple accessors to neuron's embedded objects
  protected

  def outputError
    neuron.outputError
  end

  def netInput
    neuron.netInput
  end

  def inputDistributionModel
    return neuron.inputDistributionModel
  end


end

class ForwardPropOnly < LearningStrategyBase # just forward propagation
  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = calcNetInputToNeuron
    neuron.propagateToOutput
  end
end

class LearningBP < LearningStrategyBase # strategy for standard bp learning for hidden neurons

  def startEpoch
    zeroDeltaWAccumulated
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = calcNetInputToNeuron
    neuron.propagateToOutput
  end

  def learnExample
    neuron.backPropagateFromOutputs
    neuron.error = outputError * ioDerivativeFromNetInput(netInput)
    calcDeltaWsAndAccumulate
  end

  def endEpoch
    addAccumulationToWeight
  end
end

class LearningBPOutput < LearningBP
end

class Normalization < LearningStrategyBase

  def startEpoch
    inputLinks.each { |aLink| aLink.resetAllNormalizationVariables }
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
  end

  def learnExample
    inputLinks.each { |link| link.storeEpochHistory }
  end

  def endEpoch
    inputLinks.each { |aLink| aLink.calculateNormalizationCoefficients }
  end
end

class SelfOrgStrat < LearningStrategyBase
  attr_accessor :targetMinus, :distanceBetweenTargets

  def initialize(theEnclosingNeuron, ** strategyArgs)
    super
    @targetPlus = self.findNetInputThatGeneratesMaximumOutput
    @targetMinus = -1.0 * @targetPlus
    @distanceBetweenTargets = @targetPlus - @targetMinus
  end

  def startEpoch
    zeroDeltaWAccumulated
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = calcNetInputToNeuron
    neuron.propagateToOutput
  end

  def learnExample
    neuron.error = -1.0 * neuron.ioDerivativeFromNetInput(netInput) * (((netInput - targetMinus)/distanceBetweenTargets) - 0.5)
    calcDeltaWsAndAccumulate
  end

  def endEpoch
    addAccumulationToWeight
  end
end


#########


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

class BaseModel
  include Distribution
  include Distribution::Shorthand

  # calculates the gaussian probability DENSITY function for any specified standard deviation and mean
  # the 'norm_pdf' function only properly calculates this when the standard deviation == 1.0
  # @param [real] x
  # @param [real] mean
  # @param [real] std
  def gaussPdf(x, mean, std)
    normalizedDeviationFromMean = ((x - mean) / std)
    return 0.0 if (normalizedDeviationFromMean.abs > 15.0)
    return norm_pdf(normalizedDeviationFromMean) / std # NOTE: the .abs gets rid of imaginary results in some cases
  end
end

class GaussModel < BaseModel
  attr_reader :mean, :std, :prior, :bayesNumerator

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
    @bayesNumerator = @prior * gaussPdf(inputOrOutputForExample, @mean, @std)
  end


  # called for each example
  # @param [real] bayesDenominator
  def estimateProbabilityOfExample(bayesDenominator)
    @examplesProbability = @bayesNumerator / bayesDenominator
  end

  # called for each example
  # @param [real]  inputOrOutputForExample; net summed input to neuron; or analog output of neuron
  def prepForRecalculatingModelsParams(inputOrOutputForExample)
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

#class GaussModelAdaptable < GaussModel
#
#  def initialize(mean, std, prior, numberOfExamples)
#    super
#    @newMean = nil
#    @m2 = nil
#    @delta = nil
#    self.initEpoch
#  end
#
#  def initEpoch
#    @n = 0.0
#    @newMean = 0.0
#    @m2 = 0.0
#  end
#
#  def prepForRecalculatingModelsParams(inputOrOutputForExample)
#    x = inputOrOutputForExample
#    @n += @examplesProbability
#    @delta = (x - @newMean) * @examplesProbability
#    @newMean = @newMean + (@delta / @n) if (@n > 1.0e-10)
#    deltaPrime = (x - @newMean) * @examplesProbability
#    @m2 = @m2 + (@delta * deltaPrime)
#  end
#
#  def atEpochsEndCalculateModelParams
#    @prior = @n / @numberOfExamples
#    @mean = @newMean
#    variance = @m2/(@n - 1)
#    newStd = Math.sqrt(variance)
#    @std = newStd
#  end
#
#  def to_s
#    "\t\tmean=\t#{mean}\tstd=\t#{std}\tprior=\t#{prior}\n"
#  end
#end


class GaussModelAdaptable < GaussModel

  def initialize(mean, std, prior, numberOfExamples)
    super
    @exampleNumber = nil
    @allInputs = Array.new(numberOfExamples)
    @allProbabilities = Array.new(numberOfExamples)
    self.initEpoch
  end

  def initEpoch
    @exampleNumber = 0
    @allInputs.clear
    @allProbabilities.clear
  end

  def prepForRecalculatingModelsParams(inputOrOutputForExample)
    @allInputs[@exampleNumber] = inputOrOutputForExample
    @allProbabilites[@exampleNumber] = @examplesProbability
    @exampleNumber += 1

    #@n += @examplesProbability
    #@sumSquaredProbabilities += (@examplesProbability**2)

  end

  def atEpochsEndCalculateModelParams
    puts "allInputs=\t#{@allInputs}"
    sumOfProbabilities = @allProbabilites.sum
    @mean = calcWeightedMean(sumOfProbabilities)
    @std = calcWeightedSTD(@mean)
    @prior = sumOfProbabilities / @numberOfExamples


  end

  def calcWeightedMean(sumOfWeights)
    vInputs = Vector.elements(@allInputs, false)
    @vWeights = Vector.elements(@allProbabilites, false)
    weightedSum = vInputs.inner_product(vWeights)
    weightedSum / sumOfWeights
  end

  def calcWeightedSTD(weightedMean)
    unweightedErrors = @allInputs.collect {|input| input - weightedMean}
    vUnweightedErrors = Vector.elements(unweightedErrors,false)
    vWeightedErrors = vUnweightedErrors.collect2(vWeights) {|error, weight| error * weight}
    sumSquaredErrors = vWeightedErrors.inner_product(vWeightedErrors)
  end

  def to_s
    "\t\tmean=\t#{mean}\tstd=\t#{std}\tprior=\t#{prior}\n"
  end
end


class ExampleDistributionModel < BaseModel
  attr_reader :models

  def initialize(args)
    @args = args
    @numberOfExamples = args[:numberOfExamples]
    @classesOfModels = [GaussModelAdaptable, GaussModelAdaptable, GaussModel]
    @mean = [1.0, -1.0, 0.0]
    @std = [0.5, 0.5, 4.0]
    @prior = [0.33, 0.33, 0.34]
    @models = []
  end

  # use this method at beginning of EACH META-EPOCH
  def initMetaEpoch
    numberOfModels = @classesOfModels.length
    numberOfModels.times { |i| @models << @classesOfModels[i].new(@mean[i], @std[i], @prior[i], @numberOfExamples) }
  end

  # use this method at beginning of EACH EPOCH
  def initEpoch
    @models.each { |model| model.initEpoch }
  end

  # use this method for EACH EXAMPLE
  def useExampleToImproveDistributionModel(netInputOrOutput)
    @models.each { |model| model.calculateBayesNumerator(netInputOrOutput) }
    bayesDenominator = @models.inject(0.0) { |sum, model| sum + model.bayesNumerator }
    @models.each do |model|
      model.estimateProbabilityOfExample(bayesDenominator)
      model.prepForRecalculatingModelsParams(netInputOrOutput)
    end
  end

  # use this method at the END of EACH EPOCH
  def atEpochsEndCalculateModelParams
    @models.each { |model| model.atEpochsEndCalculateModelParams }
    puts
    puts self
  end

  def calcError(netInputOrOutput)
    x = netInputOrOutput
    sumOfLikelihoods = @models.inject(0.0) { |sum, model| sum + likelihood(x, model) }
    error = @models.inject(0.0) do |sum, model|
      probabilityForModel = probabilityThatExampleCameFrom(model, x, sumOfLikelihoods)
      errorComponentForModel =  (x - model.mean) * probabilityForModel

      puts "probabilityForModel=\t#{probabilityForModel},\tnetInputOrOutput=\t#{x},\tmodel.mean=\t#{model.mean},\tmodel.std=\t#{model.std},\terrorComponentForModel=#{errorComponentForModel}"

      sum +  errorComponentForModel
    end
    puts "\t\terror=\t#{error}\n"
    error
  end

  def probabilityThatExampleCameFrom(theModel, x, sumOfLikelihoods)
    probability = likelihood(x, theModel) / sumOfLikelihoods
  end

  def likelihood(x, model)
    model.prior * gaussPdf(x, model.mean, model.std)
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
    if (neuron.inputDistributionModel.nil?)
      classOfInputDistributionModel = @strategyArgs[:classOfInputDistributionModel]
      neuron.inputDistributionModel = classOfInputDistributionModel.new(@strategyArgs) # keeping inputDistributionModel
      # in neuron for continuity across the invoking of different learning strategies.
    end
  end

  def startStrategy
    inputDistributionModel.initMetaEpoch
  end

  def startEpoch
    inputDistributionModel.initEpoch
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = calcNetInputToNeuron
    puts "netInput= #{netInput}"
    neuron.output = ioFunction(netInput)
    inputDistributionModel.useExampleToImproveDistributionModel(netInput)
  end

  def endEpoch
    inputDistributionModel.atEpochsEndCalculateModelParams
  end
end


class SelfOrgByContractingBothLobesOfDistribution < LearningStrategyBase

  def startEpoch
    zeroDeltaWAccumulated
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
  end

  def learnExample
    neuron.error = inputDistributionModel.calcError(netInput)
    calcDeltaWsAndAccumulate
  end

  def endEpoch
    addAccumulationToWeight
  end
end


#########

module ContextForLearning
  attr_accessor :learningController

  def propagate(exampleNumber)
    if learningController.output == 1.0
      super(exampleNumber)
    else
      dropOutNeuron
    end
  end

  def learnExample
    if learningController.output == 1.0
      super
    end
  end

  protected

  def dropOutNeuron
    neuron.netInput = netInput = 0.0
    neuron.output = ioFunction(netInput)
  end
end

###################################################################
###################################################################
; ######################## Learning Controllers ####################

class LearningController
  attr_accessor :sensor

  def initialize(sensor=nil)
    @sensor = sensor
  end

  def output
    1.0
  end
end

class LearningControlledByNeuron < LearningController
  def output
    transform(sensor.output)
  end

  def transform(input)
    if input >= 0.5
      1.0
    else
      0.0
    end
  end
end

class LearningControlledByNeuronNOT < LearningControlledByNeuron
  def output
    logicalNOT(transform(sensor.output))
  end

  protected

  def logicalNOT(input)
    returnValue = if input == 1.0
                    0.0
                  else
                    1.0
                  end
  end
end


