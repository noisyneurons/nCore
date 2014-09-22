### VERSION "nCore"
## ../nCore/lib/core/NeuralParts2.rb

require_relative 'NeuralParts'
############################################################


class Neuron2 < Neuron
  attr_accessor :learningStrat

  def postInitialize
    super
    @learningStrat = LearningBP.new(self) # default learner
  end

  def backPropagate
    learningStrat.backPropagate
  end
end

class OutputNeuron2 < OutputNeuron
  attr_accessor :learningStrat

  def postInitialize
    super
    @learningStrat = LearningBPOutput.new(self) # default learner
  end

  def backPropagate
    learningStrat.backPropagate
  end
end


############################################################


class LearningStrategyBase # strategy for standard bp learning for output neurons
  attr_reader :neuron, :inputLinks, :nextInChain
  include CommonNeuronCalculations

  def initialize(theEnclosingNeuron, nextInChain = nil)
    @neuron = theEnclosingNeuron
    @nextInChain = nextInChain
    @inputLinks = @neuron.inputLinks
  end
end


class LearningBP < LearningStrategyBase # strategy for standard bp learning for hidden neurons
  attr_reader :outputLinks

  def initialize(theEnclosingNeuron, nextInChain = nil)
    super
    @outputLinks = @neuron.outputLinks
  end

  def backPropagate
    neuron.error = calcNetError * neuron.ioDerivativeFromNetInput(neuron.netInput)
  end
end


class LearningBPOutput < LearningStrategyBase # strategy for standard bp learning for output neurons

  def backPropagate
    neuron.error = neuron.outputError * neuron.ioDerivativeFromNetInput(neuron.netInput)
  end
end


class LearningSelfOrg < LearningStrategyBase

  def calcSelfOrgError
    targetPlus = 2.5 # TODO need "exact number" here. -- just for illustration purposes...
    targetMinus = -1.0 * targetPlus
    distanceBetweenTargets = targetPlus - targetMinus
    neuron.error = -1.0 * neuron.ioDerivativeFromNetInput(neuron.netInput) * (((neuron.netInput - targetMinus)/distanceBetweenTargets) - 0.5)
  end

  def resetAllNormalizationVariables
    inputLinks.each { |aLink| aLink.resetAllNormalizationVariables }
  end

  def storeEpochHistory
    inputLinks.each { |link| link.storeEpochHistory }
  end

  def propagateForNormalization(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = inputLinks.inject(0.0) { |sum, link| sum + link.propagateForNormalization }
    neuron.output = neuron.ioFunction(neuron.netInput)
  end

  def calculateNormalizationCoefficients
    inputLinks.each { |aLink| aLink.calculateNormalizationCoefficients }
  end

  def afterSelfOrgReCalcLinkWeights
    biasWeight = inputLinks.inject(0.0) { |sum, link| sum + link.propagateUsingZeroInput }
    inputLinks.each { |aLink| aLink.afterSelfOrgReCalcLinkWeights }
    inputLinks[-1].weight = biasWeight
  end
end


class LearningStratContext < LearningStrategyBase
  attr_accessor :learningController

  def initialize(theEnclosingNeuron, nextInChain = nil)
    super
    @learningController = LearningController.new
  end

  def storeEpochHistory
    nextInChain.storeEpochHistory if (learningController.output == 1)
  end

  def calcDeltaWAndAccumulate
    nextInChain.calcDeltaWAndAccumulate if (learningController.output == 1)
  end

  ### forward calls to next in daisy chain below

  #def propagate
  #  nextInChain.propagate
  #end

  def backPropagate
    nextInChain.backPropagate
  end

  def calcSelfOrgError
    nextInChain.calcSelfOrgError
  end

  def propagateForNormalization(exampleNumber)
    nextInChain.propagateForNormalization(exampleNumber)
  end

  def resetAllNormalizationVariables
    nextInChain.resetAllNormalizationVariables
  end

  def calculateNormalizationCoefficients
    nextInChain.calculateNormalizationCoefficients
  end

  def afterSelfOrgReCalcLinkWeights
    nextInChain.afterSelfOrgReCalcLinkWeights
  end
end




