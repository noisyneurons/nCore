### VERSION "nCore"
## ../nCore/lib/core/NeuralParts2.rb

############################################################

module ForwardingToLearningStrategy

  def startStrategy
    learningStrat.startStrategy
  end

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

  # service routines that may be used by various learning strategies

  def calcWeightsForUNNormalizedInputs
    learningStrat.calcWeightsForUNNormalizedInputs
  end

end

class Neuron2 < Neuron
  attr_accessor :learningStrat
  include IOFunctionNotAccessibleHere
  include ForwardingToLearningStrategy

  def postInitialize
    @inputLinks = []
    @netInput = 0.0
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

########################################################################

class LinkWithNormalization < Link
  attr_accessor :inputsOverEpoch, :normalizationOffset, :largestAbsoluteArrayElement, :normalizationMultiplier

  def initialize(inputNeuron, outputNeuron, args)
    super(inputNeuron, outputNeuron, args)
    @inputsOverEpoch = []
    resetAllNormalizationVariables
  end

  def resetAllNormalizationVariables
    self.inputsOverEpoch.clear
    self.normalizationOffset = 0.0
    self.largestAbsoluteArrayElement = 1.0
    self.normalizationMultiplier = 1.0
  end

  def storeEpochHistory
    self.inputsOverEpoch << inputNeuron.output
  end

  def propagate
    return normalizationMultiplier * weight * (inputNeuron.output - normalizationOffset)
  end

  def calculateNormalizationCoefficients
    averageOfInputs = inputsOverEpoch.mean
    self.normalizationOffset = averageOfInputs
    centeredArray = inputsOverEpoch.collect { |value| value - normalizationOffset }
    largestAbsoluteArrayElement = centeredArray.minmax.abs.max.to_f
    self.normalizationMultiplier = if largestAbsoluteArrayElement > 1.0e-5
                                     1.0 / largestAbsoluteArrayElement
                                   else
                                     0.0
                                   end
  end

  def calcWeightsForUNNormalizedInputs
    self.weight = normalizationMultiplier * weight
  end

  def propagateUsingZeroInput
    return -1.0 * normalizationMultiplier * weight * normalizationOffset
  end

  def to_s
    return "Weight=\t#{weight}\tOffset=\t#{normalizationOffset}\tMultiplier=\t#{normalizationMultiplier}\tDeltaW=\t#{deltaW}\tAccumulatedDeltaW=\t#{deltaWAccumulated}\tWeightAtBeginningOfTraining=\t#{weightAtBeginningOfTraining}\tFROM: #{inputNeuron.class.to_s} #{inputNeuron.id} TO: #{outputNeuron.class.to_s} #{outputNeuron.id}"
  end
end


