### VERSION "nCore"
## ../nCore/lib/core/NeuralParts2.rb

require_relative 'NeuralParts'
############################################################

module ForwardingToLearningStrategy
  def startEpoch
    learningStrat.startEpoch
  end

  def propagate(exampleNumber)
    learningStrat.propagate(exampleNumber)
  end

  def learnExample
    learningStrat.learnExample
  end

  def endEpoch
    learningStrat.endEpoch
  end
end


class Neuron2 < Neuron
  attr_accessor :learningStrat
  include ForwardingToLearningStrategy

  def postInitialize
    super
    @learningStrat = nil
  end
end

class OutputNeuron2 < OutputNeuron
  attr_accessor :learningStrat
  include ForwardingToLearningStrategy

  def postInitialize
    super
    @learningStrat = nil
  end
end


############################################################


class LearningStrategyBase # strategy for standard bp learning for output neurons
  attr_reader :neuron, :inputLinks, :outputLinks, :nextInChain
  include CommonNeuronCalculations
  include SigmoidIOFunction

  def initialize(theEnclosingNeuron, nextInChain = nil)
    @neuron = theEnclosingNeuron
    @nextInChain = nextInChain
    @inputLinks = @neuron.inputLinks
    @outputLinks = @neuron.outputLinks if @neuron.respond_to?(:outputLinks)
  end

  def startEpoch
    zeroDeltaWAccumulated
    neuron.error = 0.0
  end

  def endEpoch
    addAccumulationToWeight
  end
end


class LearningBP < LearningStrategyBase # strategy for standard bp learning for hidden neurons

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
  end

  def learnExample
    neuron.error = calcNetError * ioDerivativeFromNetInput(neuron.netInput)
    calcDeltaWsAndAccumulate
  end
end


class LearningBPOutput < LearningStrategyBase # strategy for standard bp learning for output neurons

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron()
    neuron.output = output = ioFunction(netInput)
    neuron.target = target = neuron.arrayOfSelectedData[exampleNumber]
    neuron.outputError = output - target
  end

  def learnExample
    neuron.error = neuron.outputError * ioDerivativeFromNetInput(neuron.netInput)
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






class LearningSelfOrg < LearningStrategyBase

  def calcSelfOrgError
    targetPlus = 2.5 # TODO need "exact number" here. -- just for illustration purposes...
    targetMinus = -1.0 * targetPlus
    distanceBetweenTargets = targetPlus - targetMinus
    neuron.error = -1.0 * neuron.ioDerivativeFromNetInput(neuron.netInput) * (((neuron.netInput - targetMinus)/distanceBetweenTargets) - 0.5)
  end


  def afterSelfOrgReCalcLinkWeights
    biasWeight = inputLinks.inject(0.0) { |sum, link| sum + link.propagateUsingZeroInput }
    inputLinks.each { |aLink| aLink.afterSelfOrgReCalcLinkWeights }
    inputLinks[-1].weight = biasWeight
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




