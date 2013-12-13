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
  include SOMClusteringCode

  def postInitialize
    super
    @metricRecorder= SOMNeuronRecorder.new(self, args)
    @maxNumberOfClusteringIterations = args[:maxNumberOfClusteringIterations]
    typeOfClusterer = args[:typeOfClusterer] || DynamicClusterer
    @clusterer = typeOfClusterer.new(args)
    @clusters = @clusterer.clusters
    @dPrime = 0.0
    @trainingSequence = args[:trainingSequence]
  end

  def propagate(exampleNumber)
    super(exampleNumber)
    self.relevance = ioDerivativeFromOutput(output)
  end

  def adaptSOM
    changeInNetInputChangeDesired = determineTargetForNetInput() - netInput
    inputLinks.each { |inputLink| inputLink.adaptSOM(changeInNetInputChangeDesired) }
  end

  def determineTargetForNetInput
    closestTarget= determinePossibleTargets.min_by { |aTarget| (aTarget - netInput).abs }
  end

  def determinePossibleTargets # Can come from flocking code...
    [0.0]
  end
end


class LinkSOM < Link
  attr_accessor :somLearningRate

  def adaptSOM(changeInNetInputChangeDesired)
    self.deltaW = somLearningRate * changeInNetInputChangeDesired * inputNeuron.output
  end

end