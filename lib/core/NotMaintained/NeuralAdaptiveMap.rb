### VERSION "nCore"
## ../nCore/lib/core/NeuralAdaptiveMap.rb

require_relative 'Utilities'
require_relative 'NeuralParts'
require_relative 'SOMWeightedClustering'

module CommonNeuronCalculations
  public
  def recordResponsesForEpoch
    metricRecorder.recordResponsesForEpoch
  end
end


class NeuronSOM < Neuron
  attr_accessor :relevance, :targetNetInput

  def postInitialize
    @inputLinks = []
    @netInput = 0.0
    self.output = 0.0
    @outputLinks = []
    @exampleNumber = nil
    @metricRecorder= SOMNeuronRecorder.new(self, args)
    @trainingSequence = args[:trainingSequence]
  end

  def propagate(exampleNumber)
    self.exampleNumber = exampleNumber
    self.netInput = euclidianDistanceBetweenInputAndNeuronsWeightVector = calcNetInputToNeuron
    self.output = euclidianDistanceBetweenInputAndNeuronsWeightVector
  end

  def calcNetInputToNeuron
    sumOfSquaredDifferences = 0.0
    inputLinks.each { |link| sumOfSquaredDifferences += (link.propagate ** 2.0) }
    return (sumOfSquaredDifferences ** 0.5)
  end

  def adaptWeight(k, lambda, learningRate)
    # lambda = lambda_i * ( (lambda_f / lambda_i) ** (t / t_max) )
    h = exp( (-1.0 * k) / lambda )
    learningRate = learningRate_i * ( (learningRate_f / learningRate_i) ** (t / t_max) )
    learningRateForNeuronsProximity = learningRate * h
    inputLinks.each { |inputLink| inputLink.adaptWeight(learningRateForNeuronsProximity) }
  end

end


class LinkSOM < Link
  attr_accessor :somLearningRate

  def adaptWeight(k)

    self.deltaW = learningRate * changeInNetInputChangeDesired * inputNeuron.output
  end
end


class SOMNeuronRecorder < NeuronRecorder
  def initialize(neuron, args)

  end
end