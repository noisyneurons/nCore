### VERSION "nCore"
## ../nCore/lib/core/NeuralParts.rb

require_relative 'Utilities'

############################################################

module SigmoidIOFunction

  def ioFunction(aNetInput)
    return 1.0/(1.0 + Math.exp(-1.0 * aNetInput))
  end

  def ioDerivativeFromNetInput(aNetInput)
    return ioDerivativeFromOutput(ioFunction(aNetInput))
  end

  def ioDerivativeFromOutput(neuronsOutput)
    return (neuronsOutput * (1.0 - neuronsOutput))
  end

end

module NonMonotonicIOFunctionUnShifted

  def ioFunction(x)
    1.49786971589547 * (i(x) - 0.166192596930178)
  end

  def i(x)
    h(x,4)
  end

  def h(x, s)
    f(x) + ( -0.5 * (f(x + s) + f(x-s)) ) + 0.5
  end

  def f(x)
    1.0 / (1.0 + Math.exp(-1.0 * x))
  end

  def ioDerivativeFromNetInput(aNetInput)
    1.49786971589547 * j(aNetInput, 4.0)
  end

  def j(x, s)
    g(x, 0.0) - (0.5 * ( g(x,s) + g(x,(-1.0 * s))))
  end

  def g(x, s)
    Math.exp((-1.0 * x) + s)   /  ((Math.exp((-1.0 * x) + s))   + 1.0)  **  2.0
  end

end

module NonMonotonicIODerivative

  def ioFunction(x)
    f(x)
  end

  def f(x)
    1.0 / (1.0 + Math.exp(-1.0 * x))
  end

  def ioDerivativeFromNetInput(aNetInput)
    1.49786971589547 * j(aNetInput, 4.0)
  end

  def j(x, s)
    g(x, 0.0) - (0.5 * ( g(x,s) + g(x,(-1.0 * s))))
  end

  def g(x, s)
    Math.exp((-1.0 * x) + s)   /  ((Math.exp((-1.0 * x) + s))   + 1.0)  **  2.0
  end
end



module NonMonotonicIOFunctionShifted

  def ioFunction(x)
    (1.49786971589547 * (i(x) - 0.166192596930178)) - 0.5
  end

  def i(x)
    h(x,4)
  end

  def h(x, s)
    f(x) + ( -0.5 * (f(x + s) + f(x-s)) ) + 0.5
  end

  def f(x)
    1.0 / (1.0 + Math.exp(-1.0 * x))
  end

  def ioDerivativeFromNetInput(aNetInput)
    1.49786971589547 * j(aNetInput, 4.0)
  end

  def j(x, s)
    g(x, 0.0) - (0.5 * ( g(x,s) + g(x,(-1.0 * s))))
  end

  def g(x, s)
    Math.exp((-1.0 * x) + s)   /  ((Math.exp((-1.0 * x) + s))   + 1.0)  **  2.0
  end

end

module NonMonotonicIOFunctionOLD

  def ioFunction(aNetInput)
    h(aNetInput, 4)
  end

  def h(x, s)
    f(x) + ( -0.5 * (f(x + s) + f(x-s)) ) + 0.5
  end

  def f(x)
    1.0 / (1.0 + Math.exp(-1.0 * x))
  end

  def ioDerivativeFromNetInput(aNetInput)
    return j(aNetInput, 4.0)
  end

  def j(x, s)
    g(x, 0.0) - (0.5 * ( g(x,s) + g(x,(-1.0 * s))))
  end

  def g(x, s)
    Math.exp((-1.0 * x) + s)   /  ((Math.exp((-1.0 * x) + s))   + 1.0)  **  2.0
  end

end



module LinearIOFunction

  def ioFunction(aNetInput)
    return aNetInput
  end

  def ioDerivativeFromNetInput(aNetInput)
    return 1.0
  end

end

module SymmetricalSigmoidIOFunction

  def ioFunction(aNetInput)
    return 2.0 * (   (1.0/(1.0 + Math.exp(-1.0 * aNetInput))) - 0.5)
  end

  def ioDerivativeFromNetInput(aNetInput) # TODO speed this up.  Use sage to get the simpler analytical expression.
    return ioDerivativeFromOutput(ioFunction(aNetInput))
  end

  def ioDerivativeFromOutput(neuronsOutput)
    return 2.0 * (neuronsOutput * (1.0 - neuronsOutput))
  end

end



module CommonNeuronCalculations
  public
  attr_accessor :netInput, :inputLinks, :error, :exampleNumber, :metricRecorder


  def calcDeltaWsAndAccumulate
    inputLinks.each { |inputLink| inputLink.calcDeltaWAndAccumulate }
  end

  def addAccumulationToWeight
    inputLinks.each { |inputLink| inputLink.addAccumulationToWeight }
  end

  def zeroDeltaWAccumulated
    inputLinks.each { |inputLink| inputLink.deltaWAccumulated = 0.0 }
  end

  def recordResponsesForExample
    metricRecorder.recordResponsesForExample
  end

  def clearWithinEpochMeasures
    metricRecorder.clearWithinEpochMeasures
  end

  def learningRate=(aLearningRate)
    inputLinks.each { |aLink| aLink.learningRate = aLearningRate }
  end

  def randomizeLinkWeights
    inputLinks.each { |anInputLink| anInputLink.randomizeWeightWithinTheRange(anInputLink.weightRange) }
  end

  def calcNetInputToNeuron
    netInput = 0.0
    inputLinks.each { |link| netInput += link.propagate }
    return netInput
  end

  def calcNetError
    netError = 0.0
    outputLinks.each do |link|
      netError += link.backPropagate
    end
    return netError
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
    #@id = -1
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

class LinearNeuron < Neuron
  include LinearIOFunction
end

class LinearOutputNeuron < OutputNeuron
  include LinearIOFunction
end

#class SymmetricalNeuron < Neuron
#  include SymmetricalSigmoidIOFunction
#end
#
#class SymmetricalOutputNeuron < OutputNeuron
#  include SymmetricalSigmoidIOFunction
#end

############################################################
class Link
  attr_accessor :inputNeuron, :outputNeuron, :weightAtBeginningOfTraining,
                :learningRate, :deltaWAccumulated, :deltaW, :weightRange, :args

  def initialize(inputNeuron, outputNeuron, args)
    @inputNeuron = inputNeuron
    @outputNeuron = outputNeuron
    @args = args
    @learningRate = args[:learningRate] || 1.0
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
    #std("learningRate",learningRate)
    #std("outputNeuron.error",outputNeuron.error)
    #
    #std("inputNeuron.output",inputNeuron.output)

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
    {:neuronID => neuron.id, :netInput => neuron.netInput, :error => neuron.error, :exampleNumber => neuron.exampleNumber}
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
    {:netInput => neuron.netInput, :error => neuron.error, :exampleNumber => neuron.exampleNumber,
     :weightedErrorMetric => neuron.weightedErrorMetric}
  end
end
