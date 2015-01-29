### VERSION "nCore"
## ../nCore/lib/core/NeuralParts.rb

require_relative 'Utilities'
require_relative 'NeuralIOFunctions'
require 'forwardable'

############################################################

module CommonNeuronCalculations
  public

  def zeroDeltaWAccumulated
    inputLinks.each { |inputLink| inputLink.deltaWAccumulated = 0.0 }
  end

  def calcNetInputToNeuron
    inputLinks.inject(0.0) { |sum, link| sum + link.propagate }
  end

  def calcNetError
    outputLinks.inject(0.0) { |sum, link| sum + link.backPropagate }
  end

  def calcDeltaWsAndAccumulate
    inputLinks.each { |inputLink| inputLink.calcDeltaWAndAccumulate }
  end

  def addAccumulationToWeight
    inputLinks.each { |inputLink| inputLink.addAccumulationToWeight }
  end

  def learningRate=(aLearningRate)
    inputLinks.each { |aLink| aLink.learningRate = aLearningRate }
  end

  def randomizeLinkWeights
    inputLinks.each { |anInputLink| anInputLink.randomizeWeightWithinTheRange(anInputLink.weightRange) }
  end

  ###  standard method for with initialization

  def zeroWeights
    inputLinks.each { |inputLink| inputLink.weight = 0.0 }
  end

  def initWeights ###  only used for special form of self-org/normalization
    numberOfInputsToNeuron = inputLinks.length
    inputLinks.each do |aLink|
      verySmallNoise = 0.0001 * (rand - 0.5) # This gets rid of perfect symmetry -- which can completely stall covergence
      weight = (1.0 + verySmallNoise) / numberOfInputsToNeuron # TODO may want sqrt(numberOfInputsToNeuron)
      # weight = (0.2 + verySmallNoise) / numberOfInputsToNeuron # TODO may want sqrt(numberOfInputsToNeuron)
      aLink.weight = weight
      aLink.weightAtBeginningOfTraining = weight
    end
  end

end

############################################################
class NeuronBase
  attr_accessor :id, :args, :output, :logger
  @@ID = 0

  def NeuronBase.zeroID
    @@ID = 0
  end

  def initialize(args)
    @id = @@ID
    @@ID += 1
    @args = args
    @logger = @args[:resultsStringIOorFileIO]
    @output = 0.0
  end

  def to_s
    description = ""
    description += "\n\t#{self.class} Class; ID = #{id}\tOutput= #{output}"
  end
end

class BiasNeuron < NeuronBase #TODO should make this a singleton class!
  attr_reader :outputLinks

  def initialize(args)
    super
    @outputLinks = []
    @output = 1.0
  end

  def to_s
    description = super
    description += "\t\tNumber of Output Links=\t#{outputLinks.length}\n"
    outputLinks.each_with_index do |link, linkNumber|
      description += "\t\t\t\t\t\tOutput Link:\t#{linkNumber}\t#{link}\n"
    end
    description
  end
end

class InputNeuron < NeuronBase
  attr_accessor :outputLinks, :exampleNumber, :arrayOfSelectedData, :keyToExampleData


  def initialize(args)
    super
    @outputLinks =[]
    @exampleNumber = nil
    @arrayOfSelectedData = nil
    @keyToExampleData = :inputs
  end

  def propagate(exampleNumber)
    @exampleNumber = exampleNumber
    self.output = arrayOfSelectedData[exampleNumber]
  end

  def to_s
    description = super
    description += "\t\tNumber of Output Links=\t#{outputLinks.length}\n"
    outputLinks.each_with_index do |link, linkNumber|
      description += "\t\t\t\t\t\tOutput Link:\t#{linkNumber}\t#{link}\n"
    end
    description
  end
end

class Neuron < NeuronBase
  attr_accessor :outputLinks
  attr_accessor :netInput, :inputLinks, :error, :outputError, :exampleNumber, :metricRecorder
  include CommonNeuronCalculations
  include SigmoidIOFunction

  def initialize(args)
    super
    @inputLinks = []
    @netInput = 0.0
    @outputLinks = []
    @error = 0.0
    @outputError = nil
    @exampleNumber = nil
    # self.output = self.ioFunction(netInput) # Only doing this in case we wish to use this code for recurrent networks
  end

  def propagate(exampleNumber)
    self.exampleNumber = exampleNumber
    self.netInput = calcNetInputToNeuron()
    propagateToOutput
  end

  def propagateToOutput
    self.output = ioFunction(netInput)
  end

  def backPropagate
    backPropagateFromOutputs
    self.error = outputError * ioDerivativeFromNetInput(netInput)
  end

  def backPropagateFromOutputs
    self.outputError = calcNetError
  end

  def to_s
    description = super
    description += "Net Input=\t#{netInput}\tError=\t#{error}\tOutput Error=\t#{outputError}\n"
    description += "\t\tNumber of Input Links=\t#{inputLinks.length}\n"
    inputLinks.each_with_index do |link, linkNumber|
      description += "\t\t\t\t\t\tInput Link:\t#{linkNumber}\t#{link}\n"
    end
    description += "\t\tNumber of Output Links=\t#{outputLinks.length}\n"
    outputLinks.each_with_index do |link, linkNumber|
      description += "\t\t\t\t\t\tOutput Link:\t#{linkNumber}\t#{link}\n"
    end
    return description
  end
end

class OutputNeuron < Neuron
  attr_accessor :arrayOfSelectedData, :keyToExampleData, :target, :weightedErrorMetric

  def initialize(args)
    super
    @arrayOfSelectedData = nil
    @weightedErrorMetric = nil
    @target = nil
    @keyToExampleData = :targets
    # self.output = self.ioFunction(netInput) # Only doing this in case we wish to use this code for recurrent networks
  end

  def propagateToOutput
    self.output = ioFunction(netInput)
    self.target = arrayOfSelectedData[exampleNumber]
  end

  def backPropagateFromOutputs
    self.outputError = output - target
  end

  def calcSumOfSquaredErrors
    sumSquaredErrorForNeuron = metricRecorder.withinEpochMeasures.inject(0.0) do |sum, exampleMeasures|
      sum + exampleMeasures[:weightedErrorMetric]
    end
    return sumSquaredErrorForNeuron
  end

  def calcWeightedErrorMetricForExample
    self.weightedErrorMetric = outputError * outputError # Assumes squared error criterion
  end

  def to_s
    description = ""
    description += "\n\t#{self.class} Class; ID = #{id}\tOutput= #{output}"
    description += "\tNet Input=\t#{netInput}\tError=\t#{error}\tOutput Error=\t#{outputError}\tWeightedErrorMetric=\t#{weightedErrorMetric}\n"
    description += "\t\tNumber of Input Links=\t#{inputLinks.length}\n"
    inputLinks.each_with_index do |link, linkNumber|
      description += "\t\t\t\t\t\tInput Link:\t#{linkNumber}\t#{link}\n"
    end
    description
  end
end

###### MODULE for Forwarding To Learning Strategy for Neuron ######

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

####### Neurons with Plug-in Learning Strategies ################

class Neuron2 < Neuron
  attr_accessor :learningStrat
  include ForwardingToLearningStrategy

  def initialize(args)
    super
    @learningStrat = nil
  end

  def to_s
    description = super
    description += "IOFunction=\t#{learningStrat.strategyArgs[:ioFunction]}\n" unless (learningStrat.nil?)
    return description
  end
end

class OutputNeuron2 < OutputNeuron
  attr_accessor :learningStrat
  include ForwardingToLearningStrategy

  def initialize(args)
    super
    @learningStrat = nil
  end

  def to_s
    description = super
    description += "IOFunction=\t#{learningStrat.strategyArgs[:ioFunction]}\n" unless (learningStrat.nil?)
    return description
  end
end

class Neuron3 < Neuron2
  attr_accessor :inputDistributionModel

  def initialize(args)
    super
    @inputDistributionModel = nil
  end
end

class OutputNeuron3 < OutputNeuron2
  attr_accessor :inputDistributionModel

  def initialize(args)
    super
    @inputDistributionModel = nil
  end
end


#####################  Links & Weights     #########################


class LinkBase
  attr_accessor :inputNeuron, :outputNeuron, :args

  def initialize(inputNeuron, outputNeuron, args)
    @inputNeuron = inputNeuron
    @outputNeuron = outputNeuron
    @args = args
  end

  def to_s
    return "LinkBase\tFROM: #{inputNeuron.class.to_s} #{inputNeuron.id} TO: #{outputNeuron.class.to_s} #{outputNeuron.id}"
  end

end

class Link < LinkBase
  attr_accessor :weightAtBeginningOfTraining, :learningRate, :deltaWAccumulated, :deltaW, :weightRange, :logger

  def initialize(inputNeuron, outputNeuron, args)
    super
    @logger = @args[:resultsStringIOorFileIO]
    @learningRate = args[:learningRate]
    @weight = 0.0
    @weightRange = args[:weightRange]
    randomizeWeightWithinTheRange(weightRange)
    @deltaW = 0.0
    @deltaWAccumulated = 0.0
  end

  def weight
    case @weight
      when Float
        return @weight
      when SharedWeight
        return @weight.value
      else
        raise TypeError.new("Wrong Class for link's weight:  #{@weight.inspect} ")
    end
  end

  def weight=(someObject)
    case @weight
      when Float
        @weight = someObject
      when SharedWeight
        raise TypeError.new("Wrong Class used to set link's weight #{someObject.inspect} ") unless (someObject.class == Float)
        @weight.value = someObject
      else
        raise TypeError.new("Wrong Class for link's weight #{@weight.inspect} ")
    end
    return someObject
  end

  def calcDeltaWAndAccumulate
    self.deltaWAccumulated += calcDeltaW
  end

  def addAccumulationToWeight
    self.weight = weight - deltaWAccumulated
  end

  def propagate
    return inputNeuron.output * weight
  end

  def backPropagate
    return outputNeuron.error * weight
  end

  def calcDeltaW
    self.deltaW = learningRate * outputNeuron.error * inputNeuron.output
  end

  def weightUpdate
    self.weight = weight - @deltaW
  end

  def randomizeWeightWithinTheRange(weightRange)
    self.weight = weightRange * (rand - 0.5)
    self.weightAtBeginningOfTraining = weight
  end

  def to_s
    return "Weight=\t#{weight}\tDeltaW=\t#{deltaW}\tAccumulatedDeltaW=\t#{deltaWAccumulated}\tWeightAtBeginningOfTraining=\t#{weightAtBeginningOfTraining}\tFROM: #{inputNeuron.class.to_s} #{inputNeuron.id} TO: #{outputNeuron.class.to_s} #{outputNeuron.id}"
  end
end

class SuppressorLink < LinkBase
  attr_accessor :reverse, :disable

  def initialize(inputNeuron, outputNeuron, args)
    super
    @disable = true
    @reverse = false
  end

  # This could be stochastic, with probability of suppression a function
  # of the input neuron's output
  def suppress?
    return false if(disable)
    returnValue = transform(@inputNeuron.output)
    case @reverse
      when false
        returnValue
      when true
        !returnValue
    end
  end

  def transform(input)
    if input >= 0.5
      true
    else
      false
    end
  end


  def to_s
    return "SuppressorLink\tSuppress: #{self.suppress?}\tFROM: #{inputNeuron.class.to_s} #{inputNeuron.id} TO: #{outputNeuron.class.to_s} #{outputNeuron.id}"
  end
end

### Link for normalization of inputs

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

class SharedWeight
  attr_accessor :value

  def initialize(aValueForTheWeight)
    @value = aValueForTheWeight
  end
end

############