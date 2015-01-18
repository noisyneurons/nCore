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

class Model
  attr_accessor :mean, :std, :inclusionProbability

  def initialize(mean, std, inclusionProbability)
    @mean = mean
    @std = std
    @inclusionProbability = inclusionProbability
  end

end

class InputDistributionSymmetrical
  attr_accessor :models

  def initialize(strategyArgs)
    @mean = 1.0
    @std = 3.0
    @inclusionProbability = 0.5
    @models = []
    self.createInitialModels
  end

  def createInitialModels
    @models <<  GaussModel.new(@mean, @std, @inclusionProbability)
    @models <<  GaussModel.new((-1.0 * @mean), @std, (1.0 - @inclusionProbability) )
    @models <<  UniformModel.new(bottom=-4.0, top=4.0)
  end

  def startNextIterationToImproveModel
    thetaModelN = {:mean => (-1.0 * revisedMean), :std => revisedStd}


  end

  def weightAndIncludeExample(netInput)


    posteriorProbabilityThatExampleIsFromModelA =  (likelihoodForModelA * aPriorityA) /
      ( (likelihoodForModelA * aPriorityA) + (likelihoodForModelB * aPriorityB) + (likelihoodForModelC * aPriorityC) )



   # models.collect

  end

  def calculateModelParams
    return
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
    inputDistributionModel.weightAndIncludeExample(netInput)
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


