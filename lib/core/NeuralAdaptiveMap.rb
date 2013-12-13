### VERSION "nCore"
## ../nCore/lib/core/NeuralAdaptiveMap.rb

require_relative 'Utilities'
require_relative 'NeuralParts'

module CommonNeuronCalculations
  public
end


class NeuronSOM < Neuron
  attr_accessor :relevance, :targetNetInput

  def propagate(exampleNumber)
    super(exampleNumber)
    self.relevance = ioDerivativeFromOutput(output)
  end

  def adaptSOM
    changeInNetInputChangeDesired = determineTargetForNetInput() - netInput
    inputLinks.each { |inputLink| inputLink.adaptSOM(changeInNetInputChangeDesired) }
  end

  def determineTargetForNetInput
    findClosestTarget(determinePossibleTargets())
  end

  def determinePossibleTargets
    [0.0]
  end

  def findClosestTarget(possibleTargets)
    closestTarget = nil
    minimumDistanceToTarget = 1e100
    possibleTargets.each do |aTarget|
      distanceToTarget = (aTarget - netInput).abs
      if (distanceToTarget < minimumDistanceToTarget)
        minimumDistanceToTarget = distanceToTarget
        closestTarget = aTarget
      end
    end
    closestTarget
  end

end


class LinkSOM < Link
  attr_accessor :somLearningRate

  def adaptSOM(changeInNetInputChangeDesired)
    self.deltaW = somLearningRate * changeInNetInputChangeDesired * inputNeuron.output
  end

end