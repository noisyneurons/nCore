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

  def finishLearningStrategy
    learningStrat.finishLearningStrategy
  end
end


class Neuron2 < Neuron
  attr_accessor :learningStrat
  include IOFunctionNotAccessibleHere
  include ForwardingToLearningStrategy

  def postInitialize
    @inputLinks = []
    @netInput = 0.0
    #self.output = self.ioFunction(@netInput) # Only doing this in case we wish to use this code for recurrent networks
    @outputLinks = []
    @error = 0.0
    @exampleNumber = nil
    @learningStrat = nil
  end

end

class OutputNeuron2 < OutputNeuron
  attr_accessor :learningStrat
  include IOFunctionNotAccessibleHere
  include ForwardingToLearningStrategy

  def postInitialize
    @netInput = 0.0
    # self.output = ioFunction(netInput) # Only doing this in case we wish to use this code for recurrent networks
    @inputLinks = []
    @error = 0.0
    @outputError = nil
    @arrayOfSelectedData = nil
    @exampleNumber = nil
    @weightedErrorMetric = nil
    @target = nil
    @keyToExampleData = :targets
    @learningStrat = nil
  end
end


############################################################


class LearningStrategyBase # strategy for standard bp learning for output neurons
  attr_reader :neuron, :inputLinks, :outputLinks
  include CommonNeuronCalculations

  def initialize(theEnclosingNeuron)
    @neuron = theEnclosingNeuron
    @inputLinks = @neuron.inputLinks
    @outputLinks = @neuron.outputLinks if @neuron.respond_to?(:outputLinks)
  end

  def startStrategy
    neuron.output = ioFunction(neuron.netInput) # Probably only need to do this in special cases, like simulating recurrent nets
  end
end


class LearningBP < LearningStrategyBase # strategy for standard bp learning for hidden neurons

  def startEpoch
    zeroDeltaWAccumulated
    error = 0.0
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
  end

  def learnExample
    neuron.error = calcNetError * ioDerivativeFromNetInput(neuron.netInput)
    calcDeltaWsAndAccumulate
  end

  def endEpoch
    addAccumulationToWeight
  end
end


class LearningBPOutput < LearningStrategyBase # strategy for standard bp learning for output neurons

  def startEpoch
    zeroDeltaWAccumulated
    error = 0.0
  end

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

  def endEpoch
    addAccumulationToWeight
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


#class LearningSelfOrg < LearningStrategyBase
#
#  def calcSelfOrgError
#    targetPlus = 2.5 # TODO need "exact number" here. -- just for illustration purposes...
#    targetMinus = -1.0 * targetPlus
#    distanceBetweenTargets = targetPlus - targetMinus
#    neuron.error = -1.0 * neuron.ioDerivativeFromNetInput(neuron.netInput) * (((neuron.netInput - targetMinus)/distanceBetweenTargets) - 0.5)
#  end
#
#
#  def afterSelfOrgReCalcLinkWeights
#    biasWeight = inputLinks.inject(0.0) { |sum, link| sum + link.propagateUsingZeroInput }
#    inputLinks.each { |aLink| aLink.afterSelfOrgReCalcLinkWeights }
#    inputLinks[-1].weight = biasWeight
#  end
#
#  def afterSelfOrgReCalcLinkWeights
#    biasWeight = inputLinks.inject(0.0) { |sum, link| sum + link.propagateUsingZeroInput }
#    inputLinks.each { |aLink| aLink.afterSelfOrgReCalcLinkWeights }
#    inputLinks[-1].weight = biasWeight
#  end
#
#
#end


class LearningStratContext < LearningStrategyBase
  attr_accessor :learningController

  def initialize(theEnclosingNeuron)
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




