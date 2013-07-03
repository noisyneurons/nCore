### VERSION "nCore"
## ../nCore/lib/core/ExampleImportanceMods.rb
# This code modifies the WeightedClustering.rb and NeuralPartsExtended.rb code in order to incorporate a simple
# form of example importance processing

require_relative 'Utilities'

module CommonNeuronCalculations
  def examplesImportance(netInput)
    ioDerivativeFromNetInput(netInput)
  end
end

class Cluster # Mods for calculation of cluster's center, given each examples importance.
  include CommonNeuronCalculations

  def calcCenterInVectorSpace(examples)
    sumOfWeightedExamples = sumUpExamplesWeightedByMembershipAndImportanceInThisCluster(examples)
    self.center = sumOfWeightedExamples / sumTheWeightsTimesTheExamplesImportance(examples)
  end

  def sumUpExamplesWeightedByMembershipAndImportanceInThisCluster(examples)
    sum = Vector.elements(Array.new(examplesVectorLength, 0.0), copy=false)
    membershipWeightForEachExample.each_with_index do |anExampleWeight, indexToExample|
      example = examples[indexToExample]
      sum += (examplesImportance(example[0]) * anExampleWeight**m * example)
    end
    return sum
  end

  def sumTheWeightsTimesTheExamplesImportance(examples)
    sum = 0.0
    membershipWeightForEachExample.each_with_index do |anExampleWeight, indexToExample|
      example = examples[indexToExample]
      sum += (examplesImportance(example[0]) * anExampleWeight**m)
    end
    return sum
  end

end

module CommonClusteringCode
  include CommonNeuronCalculations

  # *** This function should not be called before an entire batch has been processed by the clusterer ***
  def calcLocalFlockingError
    clusters_center_virtual_or_exact = yield
    distanceToWeightedExamplesCenter = clusters_center_virtual_or_exact - locationOfExample
    exampleImportanceFactor = exampleImportanceCorrectionFactor(clusters_center_virtual_or_exact, locationOfExample)
    self.localFlockingError = exampleImportanceFactor * distanceToWeightedExamplesCenter
    self.accumulatedAbsoluteFlockingError += localFlockingError.abs
    return localFlockingError
  end

  def exampleImportanceCorrectionFactor(clustersCenter, locationOfExample)
    factor = examplesImportance(locationOfExample) / examplesImportance(clustersCenter)
  end
end