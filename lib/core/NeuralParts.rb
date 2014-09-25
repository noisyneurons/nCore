### VERSION "nCore"
## ../nCore/lib/core/NeuralParts.rb

require_relative 'Utilities'
require_relative 'NeuralIOFunctions'
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

  def zeroWeights
    inputLinks.each { |inputLink| inputLink.weight = 0.0 }
  end
end

############################################################
class NeuronBase
  attr_accessor :id, :args, :output
  @@ID = 0

  def NeuronBase.zeroID
    @@ID = 0
  end

  def initialize(args)
    @id = @@ID
    @@ID += 1
    @args = args
    @output = 0.0
    postInitialize
  end

  def to_s
    description = ""
    description += "\n\t#{self.class} Class; ID = #{id}\tOutput= #{output}"
  end
end

class BiasNeuron < NeuronBase #TODO should make this a singleton class!
  attr_reader :outputLinks

  def postInitialize
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
  attr_accessor :outputLinks, :arrayOfSelectedData, :keyToExampleData

  def postInitialize
    @outputLinks =[]
    @arrayOfSelectedData = nil
    @keyToExampleData = :inputs
  end

  def propagate(exampleNumber)
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
  attr_accessor :netInput, :inputLinks, :error, :exampleNumber, :metricRecorder
  include CommonNeuronCalculations
  include SigmoidIOFunction

  def postInitialize
    @inputLinks = []
    @netInput = 0.0
    self.output = self.ioFunction(@netInput) # Only doing this in case we wish to use this code for recurrent networks
    @outputLinks = []
    @error = 0.0
    @exampleNumber = nil
    @metricRecorder= NeuronRecorder.new(self, args)
  end

  def propagate(exampleNumber)
    self.exampleNumber = exampleNumber
    self.netInput = calcNetInputToNeuron
    self.output = ioFunction(netInput)
  end

  def backPropagate
    self.error = calcNetError * ioDerivativeFromNetInput(netInput)
  end

  def to_s
    description = super
    description += "Net Input=\t#{netInput}\tError=\t#{error}\n"
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

class OutputNeuron < NeuronBase
  attr_accessor :arrayOfSelectedData, :keyToExampleData, :target, :outputError, :weightedErrorMetric
  attr_accessor :netInput, :inputLinks, :error, :exampleNumber, :metricRecorder
  include CommonNeuronCalculations
  include SigmoidIOFunction

  def postInitialize
    @netInput = 0.0
    self.output = ioFunction(netInput) # Only doing this in case we wish to use this code for recurrent networks
    @inputLinks = []
    @error = 0.0
    @outputError = nil
    @arrayOfSelectedData = nil
    @exampleNumber = nil
    @weightedErrorMetric = nil
    @target = nil
    @keyToExampleData = :targets
    @metricRecorder = OutputNeuronRecorder.new(self, args)
  end

  def propagate(exampleNumber)
    self.exampleNumber = exampleNumber
    self.netInput = calcNetInputToNeuron()
    self.output = ioFunction(netInput)
    self.target = arrayOfSelectedData[exampleNumber]
    self.outputError = output - target
  end

  def backPropagate
    self.error = outputError * ioDerivativeFromNetInput(netInput)
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
    description = super
    description += "Net Input=\t#{netInput}\tError=\t#{error}\tWeightedErrorMetric=\t#{weightedErrorMetric}\n"
    description += "\t\tNumber of Input Links=\t#{inputLinks.length}\n"
    inputLinks.each_with_index do |link, linkNumber|
      description += "\t\t\t\t\t\tInput Link:\t#{linkNumber}\t#{link}\n"
    end
    description
  end
end

class NoisyNeuron < Neuron
  attr_accessor :probabilityOfBeingEnabled, :enabled, :outputWhenNeuronDisabled, :learning

  def postInitialize
    super
    @learning = true
    @probabilityOfBeingEnabled = [args[:probabilityOfBeingEnabled], 0.01].max
    @outputWhenNeuronDisabled = self.ioFunction(0.0)
  end

  def propagate(exampleNumber)
    case
      when learning == true
        self.output = if (self.enabled = rand < probabilityOfBeingEnabled)
                        super(exampleNumber)
                      else
                        outputWhenNeuronDisabled
                      end

      when learning == false
        signal = super(exampleNumber) - outputWhenNeuronDisabled
        self.output = (signal / probabilityOfBeingEnabled) + outputWhenNeuronDisabled
      else
        STDERR.puts "error, 'learning' variable not set to true or false!!"
    end
    output
  end

  def backPropagate
    self.error = if (enabled)
                   super
                 else
                   0.0
                 end
  end
end

############################################################
class Link
  attr_accessor :inputNeuron, :outputNeuron, :weightAtBeginningOfTraining,
                :learningRate, :deltaWAccumulated, :deltaW, :weightRange, :args

  def initialize(inputNeuron, outputNeuron, args)
    @inputNeuron = inputNeuron
    @outputNeuron = outputNeuron
    @args = args
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

class SharedWeight
  attr_accessor :value

  def initialize(aValueForTheWeight)
    @value = aValueForTheWeight
  end
end

############################################################      N
class NeuronRecorder
  attr_accessor :neuron, :args, :withinEpochMeasures

  def initialize(neuron, args = {})
    @neuron = neuron
    @args = args
    @withinEpochMeasures = []
  end

  def dataToRecord
    {:neuronID => neuron.id, :netInput => neuron.netInput, :output => neuron.output, :error => neuron.error, :exampleNumber => neuron.exampleNumber}
  end

  def recordResponsesForExample
    self.withinEpochMeasures << dataToRecord
    return dataToRecord
  end

  def clearWithinEpochMeasures
    withinEpochMeasures.clear
  end
end

class OutputNeuronRecorder < NeuronRecorder
  def dataToRecord
    {:neuronID => neuron.id, :netInput => neuron.netInput, :output => neuron.output, :error => neuron.error, :exampleNumber => neuron.exampleNumber,
     :weightedErrorMetric => neuron.weightedErrorMetric}
  end
end
