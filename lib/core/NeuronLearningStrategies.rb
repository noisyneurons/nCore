### VERSION "nCore"
; ## ../nCore/lib/core/NeuronLearningStrategies.rb

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
    self.extend(ioFunction)
    neuron.extend(ioFunction)
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

  def netInput
    neuron.netInput
  end

  def netInput=(aValue)
    neuron.netInput = aValue
  end

  def inputDistributionModel
    return neuron.inputDistributionModel
  end

end

class ForwardPropOnly < LearningStrategyBase # just forward propagation
  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    self.netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
  end
end

class LearningBP < LearningStrategyBase # strategy for standard bp learning for hidden neurons

  def startEpoch
    zeroDeltaWAccumulated
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    self.netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
  end

  def learnExample
    neuron.error = calcNetError * ioDerivativeFromNetInput(netInput)
    calcDeltaWsAndAccumulate
  end

  def endEpoch
    addAccumulationToWeight
  end
end

class LearningBPOutput < LearningBP

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    self.netInput = calcNetInputToNeuron()
    neuron.output = output = ioFunction(netInput)
    neuron.target = target = neuron.arrayOfSelectedData[exampleNumber]
    neuron.outputError = output - target
  end

  def learnExample
    neuron.error = neuron.outputError * ioDerivativeFromNetInput(netInput)
    calcDeltaWsAndAccumulate
  end
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
    self.netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
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

  def initialize(mean, std, prior, numberOfExamples)
    @numberOfExamples = numberOfExamples
    @mean = mean
    @std = std
    @prior = prior
    @bayesNumerator = nil
    @examplesProbability = nil

    @n = nil
    @newMean = nil
    @m2 = nil
    @delta = nil

    reInitialize
  end

  def reInitialize
    @n = 0.0
    @newMean = 0.0
    @m2 = 0.0
  end

  def calculateBayesNumerator(neuronsInputOrOutput)
    @bayesNumerator = Distribution::Normal.pdf(neuronsInputOrOutput) * @prior
  end

  def estimateProbabilityOfExample(bayesDenominator)
    @examplesProbability = @bayesNumerator / bayesDenominator
  end

# Had to Modify this for weighting examples according to estimated probability that example came from the model.
#  def prepForRecalculatingModelsParams(x)
#    @effectiveNumberOfExamplesRepresentedByModel += @examplesProbability
#    @weightedX = @examplesProbability * x
#
#    @n = @n + 1
#    @delta = x - @newMean
#    @newMean = @newMean + (@delta / @n)
#    @m2 = @m2 + (@delta*(x - @newMean))
#  end
#
#  def recalculateModelsParamsAtEndOfEpoch
#    @mean = @newMean
#
#    variance = @m2/(@n - 1)
#    newStd = Math.sqrt(variance)
#    @std = newStd
#
#    @prior = @effectiveNumberOfExamplesRepresentedByModel / @numberOfExamples
#  end


  def prepForRecalculatingModelsParams(x)
    @n += @examplesProbability
    @delta = (x - @newMean) * @examplesProbability
    @newMean = @newMean + (@delta / @n)
    deltaPrime = (x - @newMean) * @examplesProbability
    @m2 = @m2 + (@delta * deltaPrime)
  end

  def recalculateModelsParamsAtEndOfEpoch
    @mean = @newMean

    variance = @m2/(@n - 1)
    newStd = Math.sqrt(variance)
    @std = newStd

    @prior = @n / @numberOfExamples
  end
end

#  Using a mixture of simple models to model the "changing" distribution of the examples within a given neuron
#    posteriorProbabilityThatExampleIsFromModelA = (likelihoodForModelA * aPriorityA) /
#        ((likelihoodForModelA * aPriorityA) + (likelihoodForModelB * aPriorityB) + (likelihoodForModelC * aPriorityC))
class ExampleDistributionModel
  attr_accessor :models

  def initialize(args)
    @args = args
    @numberOfExamples = args[:numberOfExamples]
    @mean = [1.0, -1.0, 0.0]
    @std = [3.0, 3.0, 9.0]
    @prior = [0.333, 0.333, 0.334]
    @models = []
    self.createInitialModels
  end

  def createInitialModels
    numberOfModels = @mean.length
    numberOfModels.times { |i| @models << GaussModel.new(@mean[i], @std[i], @prior[i], @numberOfExamples) }
  end

  def startNextIterationToImproveModel
    @models.each { |model| model.reInitialize }
  end

  def useExampleToImproveDistributionModel(netInputOrOutput)

    @models.each { |model| model.calculateBayesNumerator(netInputOrOutput) }
    bayesDenominator = @models.inject(0.0) { |sum, model| sum + model.bayesNumerator }
    @models.each do |model|
      model.estimateProbabilityOfExample(bayesDenominator)
      model.prepForRecalculatingModelsParams(netInputOrOutput)
    end
  end

  def calculateModelParams
    @models.each { |model| model.recalculateModelsParamsAtEndOfEpoch }
  end

  def calcError(netInput)
  end
end

class EstimateInputDistribution < LearningStrategyBase

  def initialize(theEnclosingNeuron, ** strategyArgs)
    super
    classOfInputDistributionModel = @strategyArgs[:classOfInputDistributionModel]
    neuron.inputDistributionModel = classOfInputDistributionModel.new(@strategyArgs) # keeping inputDistributionModel
    # in neuron for continuity across the invoking of different learning strategies.
  end

  def startStrategy
    inputDistributionModel.createInitialModel
  end

  def startEpoch
    inputDistributionModel.startNextIterationToImproveModel
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    self.netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
    inputDistributionModel.useExampleToImproveDistributionModel(netInputOrOutput = netInput)
  end

  def endEpoch
    inputDistributionModel.calculateModelParams
  end

end

class SelfOrgByContractingBothLobesOfDistribution < LearningStrategyBase

  def startEpoch
    zeroDeltaWAccumulated
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    self.netInput = calcNetInputToNeuron
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


