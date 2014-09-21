### VERSION "nCore"
## ../nCore/lib/core/NeuralParts2.rb

require_relative 'NeuralParts'
############################################################


class Neuron2 < Neuron
  attr_accessor :learningStrat

  def postInitialize
    super
    @learningStrat = LearningBP.new(self)
  end

  def propagate(exampleNumber)
    learningStrat.propagate(exampleNumber)
  end

  def backPropagate
    learningStrat.backPropagate
  end
end

class OutputNeuron2 < OutputNeuron
  attr_accessor :learningStrat

  def postInitialize
    super
    @learningStrat = LearningBPOutput.new(self)
  end

  def propagate(exampleNumber)
    learningStrat.propagate(exampleNumber)
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

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = calcNetInputToNeuron
    neuron.output = neuron.ioFunction(neuron.netInput)
  end

  def backPropagate
    neuron.error = calcNetError * neuron.ioDerivativeFromNetInput(neuron.netInput)
  end
end



class LearningBPOutput < LearningStrategyBase  # strategy for standard bp learning for output neurons

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = calcNetInputToNeuron
    neuron.output = neuron.ioFunction(neuron.netInput)
    neuron.target = neuron.arrayOfSelectedData[exampleNumber]
    neuron.outputError = neuron.output - neuron.target
  end

  def backPropagate
    neuron.error = neuron.outputError * neuron.ioDerivativeFromNetInput(neuron.netInput)
  end
end

