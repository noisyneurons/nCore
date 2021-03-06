### VERSION "nCore"
; ## ../nCore/lib/core/NeuronLearningStrategies.rb

require 'distribution'
###################################################################
###################################################################
; ####################### Basic Learning Strategies ############################
;
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

  def endStrategy
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


#;
; ####################### Version 0.2 Learning Strategies #######################
;
; #---- Support Classes: Distribution Models (and sub-models)
;
# base Gaussian model for estimating a single gaussian's parameters
# This base model is only useful as part of a mixture of models, where some
# models parameters are "more adaptable."
# A Gaussian model to incorporate into a "mixture of models" distribution model
# of a neuron's netInputs or outputs across examples
#
# See MiscellaneousCode.rb for iterative/streaming/on-line method for single pass variance estimation:
# FROM  Wikipedia  http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
class BaseModel
  include Distribution
  include Distribution::Shorthand

  def startStrategy;
  end

  # calculates the gaussian probability DENSITY function for any specified standard deviation and mean
  # the 'norm_pdf' function only properly calculates this when the standard deviation == 1.0
  # TODO may want to use LogGaussian instead!!!!
  # @param [real] x
  # @param [real] mean
  # @param [real] std
  def gaussPdf(x, mean, std)
    std = 1e-15 if (std < 1e-15) # 1e-20 does NOT work.  You get NaN in simulation results
    normalizedDeviationFromMean = (x - mean) / std
    return norm_pdf(normalizedDeviationFromMean) / std # NOTE: the .abs gets rid of imaginary results in some cases
  end
end

class GaussModel < BaseModel
  attr_accessor :mean, :std
  attr_reader :prior, :bayesNumerator

  def initialize(mean, std, prior)
    @mean = mean
    @std = std
    @prior = prior
    @bayesNumerator = nil
    @examplesProbability = nil
    @sumOfProbabilities = 0.0
    @exampleNumber = 0
  end

  # called at the beginning of each epoch
  def initEpoch
    @sumOfProbabilities = 0.0
    @exampleNumber = 0
  end

  # called for each example
  # @param [real] inputOrOutputForExample net summed input to neuron; or analog output of neuron
  def calculateBayesNumerator(inputOrOutputForExample)
    numerator = @prior * gaussPdf(inputOrOutputForExample, @mean, @std)
    floorForNumerator =  1.0e-10
    numerator = floorForNumerator if (numerator < floorForNumerator)
    @bayesNumerator = numerator
  end

  # called for each example
  # @param [real] bayesDenominator
  def estimateProbabilityOfExample(bayesDenominator)
    @examplesProbability = @bayesNumerator / bayesDenominator
  end

  # called for each example
  # @param [real]  inputOrOutputForExample; net summed input to neuron; or analog output of neuron
  def prepForRecalculatingModelsParams(inputOrOutputForExample)
    @exampleNumber += 1
    @sumOfProbabilities += @examplesProbability
  end

  # called at the end of each epoch
  def atEpochsEndCalculateModelParams
    @prior = @sumOfProbabilities / @exampleNumber
  end

  def to_s
    "\t\tmean=\t#{mean}\tstd=\t#{std}\tprior=\t#{prior}\n"
  end
end

class GaussModelAdaptable < GaussModel
  def initialize(mean, std, prior)
    super
    @allInputs = []
    @allProbabilities = []
  end

  def startStrategy
    @allInputs.clear
    @allProbabilities.clear
  end

  def prepForRecalculatingModelsParams(inputOrOutputForExample)
    @allInputs[@exampleNumber] = inputOrOutputForExample
    @allProbabilities[@exampleNumber] = @examplesProbability
    @exampleNumber += 1
  end

  def atEpochsEndCalculateModelParams
    sumOfProbabilities = @allProbabilities.sum
    @mean = calcWeightedMean(sumOfProbabilities)
    @std = calcWeightedSTD(@mean)
    @prior = sumOfProbabilities / @exampleNumber # which represents the total number of examples processed by this neuron
  end

  def calcWeightedMean(sumOfProbabilities)
    weightedSum = @allInputs.to_v.inner_product(@allProbabilities.to_v)
    @mean = weightedSum / sumOfProbabilities
  end

  def calcWeightedSTD(weightedMean)
    weights = @allProbabilities.to_v
    unweightedErrors = (@allInputs.to_v).collect { |input| input - weightedMean }
    weightedErrors = (unweightedErrors.collect2(weights) { |error, weight| error * weight }).to_v
    sumSquareWeightedErrors = weightedErrors.inner_product(weightedErrors)
    sumSquaredWeights = weights.inner_product(weights)
    @std = Math.sqrt(sumSquareWeightedErrors / sumSquaredWeights)
  end

  def to_s
    "\t\tmean=\t#{mean}\tstd=\t#{std}\tprior=\t#{prior}\n"
  end
end

class ExampleDistributionModel < BaseModel
  attr_reader :models

  def initialize(args)
    @args = args
    @classesOfModels = [GaussModelAdaptable, GaussModelAdaptable, GaussModel]
    @mean = [1.0, -1.0, 0.0] # [0.026, -0.026, 0.0]
    @std = [0.5, 0.5, 4.0] # [0.01, 0.01, 0.4]
    @prior = [0.33, 0.33, 0.34]
    @models = []
    @classesOfModels.each_with_index { |classOfModel, i| @models << classOfModel.new(@mean[i], @std[i], @prior[i]) }
  end

  def startStrategy
    @models.each { |model| model.startStrategy }
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

  # Key calculation for flocking/clustering examples' netinputs
  # TODO May want to restrict calculation to the 1st 2 models; or, in other cases, for "deflocking" we may want to only
  # calculate error for the 3rd model (i.e., the model with 'always' zero mean)
  def calcError(netInputOrOutput)
    x = netInputOrOutput
    sumOfLikelihoods = @models.inject(0.0) { |sum, model| sum + likelihood(x, model) }
    error = @models.inject(0.0) do |sum, model|
      probabilityForModel = probabilityThatExampleCameFrom(model, x, sumOfLikelihoods)
      errorComponentForModel = (x - model.mean) * probabilityForModel
      sum + errorComponentForModel
    end
    error
  end

  def makeMeansOfFirst2ModelsSymmetrical
    algebraicDistanceBetweenMeans = @models[1].mean - @models[0].mean
    averageAlgebraicMean = algebraicDistanceBetweenMeans / 2.0
    @models[1].mean = averageAlgebraicMean
    @models[0].mean = -1.0 * averageAlgebraicMean
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
;
; #---- 0.2 Learning Strategies
;
class NormalizeByZeroingSumOfNetInputs < LearningStrategyBase

  def startEpoch
    @sumOfNetInputs = 0.0
    @sumOfBiasWeightsContributionToNetInput = 0.0
    @biasWeight = inputLinks[-1].weight
    @numExamplesCounted = 0
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron
    # puts "netInputAtNormalization = \t#{netInput}"
    @sumOfNetInputs += netInput
    @sumOfBiasWeightsContributionToNetInput += @biasWeight
    @numExamplesCounted += 1
  end

  def endEpoch
    biasWeight = inputLinks[-1].weight
    sumOfNetInputsWithOutBiasContribution = @sumOfNetInputs - @sumOfBiasWeightsContributionToNetInput
    inputLinks[-1].weight = -1.0 * (sumOfNetInputsWithOutBiasContribution / @numExamplesCounted)
    puts "@numExamplesCounted=\t#{@numExamplesCounted}"
  end
end

class ScaleNeuronWeights < LearningStrategyBase

  def startEpoch
    @numExamplesCounted = 0
    @largestNetInput = 0.0
    @desiredMaxNetInput = 1.0
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron
    v = netInput.abs
    @largestNetInput = v if (v > @largestNetInput)
  end

  def endEpoch
    scaleFactor = @desiredMaxNetInput / @largestNetInput
    inputLinks.each { |link| link.weight = scaleFactor *link.weight }
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
    inputDistributionModel.startStrategy
  end

  def startEpoch
    inputDistributionModel.initEpoch
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = calcNetInputToNeuron
    # puts "netInputAtEstimation = \t#{netInput}"
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
    inputDistributionModel.makeMeansOfFirst2ModelsSymmetrical
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

class MoveLobesApart < LearningStrategyBase

  def startEpoch
    zeroDeltaWAccumulated

    # determining adjustment factor necessary to obtain desired mean netInputs
    inputDistributionModel.makeMeansOfFirst2ModelsSymmetrical
    currentMeanNetInputForModel0 = inputDistributionModel.models[0].mean
    desiredMeanNetInput = @strategyArgs[:desiredMeanNetInput]
    scaleFactor = (desiredMeanNetInput / currentMeanNetInputForModel0).abs

    # also adjusting distribution component models
    @inputLinks.each { |link| link.weight = scaleFactor * link.weight }
    # also adjusting distribution component models
    inputDistributionModel.models[0].mean = scaleFactor * inputDistributionModel.models[0].mean
    inputDistributionModel.models[1].mean = scaleFactor * inputDistributionModel.models[1].mean
    inputDistributionModel.models[0].std = scaleFactor * inputDistributionModel.models[0].std
    inputDistributionModel.models[1].std = scaleFactor * inputDistributionModel.models[1].std
  end

end
;
; ####################### Module for Modulation of Learning #######################
;
module LearningSuppressionViaLink
  def propagate(exampleNumber)
    if neuron.suppressorLink.suppress?
      neuronDoesNotRespond
    else
      super(exampleNumber)
    end
  end

  def learnExample
    super unless neuron.suppressorLink.suppress?
  end

  def neuronDoesNotRespond
    neuron.netInput = netInput = 0.0
    neuron.output = ioFunction(netInput)
  end
end
;
; ####################### Module used by Learning Controllers #######################
;
#module ContextForLearning
#  attr_accessor :learningController
#
#  def propagate(exampleNumber)
#    if learningController.output == 1.0
#      super(exampleNumber)
#    else
#      dropOutNeuron
#    end
#  end
#
#  def learnExample
#    if learningController.output == 1.0
#      super
#    end
#  end
#
#  protected
#
#  def dropOutNeuron
#    neuron.netInput = netInput = 0.0
#    neuron.output = ioFunction(netInput)
#  end
#end
#;
#; ######################## Learning Controllers ####################################
#;
#class LearningController
#  attr_accessor :sensor
#
#  def initialize(sensor=nil)
#    @sensor = sensor
#  end
#
#  def output
#    1.0
#  end
#end
#
#class LearningControlledByNeuron < LearningController
#  def output
#    transform(sensor.output)
#  end
#
#  def transform(input)
#    if input >= 0.5
#      1.0
#    else
#      0.0
#    end
#  end
#end
#
#class LearningControlledByNeuronNOT < LearningControlledByNeuron
#  def output
#    logicalNOT(transform(sensor.output))
#  end
#
#  protected
#
#  def logicalNOT(input)
#    returnValue = if input == 1.0
#                    0.0
#                  else
#                    1.0
#                  end
#  end
#end
