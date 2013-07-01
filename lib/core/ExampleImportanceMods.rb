### VERSION "nCore"
## ../nCore/lib/core/ExampleImportanceMods.rb
# This code modifies the WeightedClustering.rb and NeuralPartsExtended.rb code in order to incorporate a simple
# form of example importance processing


require_relative 'Utilities'

class Cluster   # TODO still do not understand the small difference in outcome for these two versions of Cluster!
  include CommonNeuronCalculations

  def calcCenterInVectorSpace(examples)
    sumOfWeightedExamples = sumUpExamplesWeightedByMembershipInThisCluster(examples)
    self.center = sumOfWeightedExamples / sumTheWeightsTimesTheExamplesImportance(examples)
  end

  def sumUpExamplesWeightedByMembershipInThisCluster(examples)
    sum = Vector.elements(Array.new(examplesVectorLength, 0.0), copy=false)
    membershipWeightForEachExample.each_with_index do |anExampleWeight, indexToExample|
      example = examples[indexToExample]
      sum += ( examplesImportance(example) * anExampleWeight**m  * example )
    end
    return sum
  end

  def sumTheWeightsTimesTheExamplesImportance(examples)
    sum = 0.0
    membershipWeightForEachExample.each_with_index do |anExampleWeight, indexToExample|
      example = examples[indexToExample]
      sum += ( examplesImportance(example) * anExampleWeight**m )
    end
    return sum
  end

  def examplesImportance(example)
    netInput = example[0]
    ioDerivativeFromNetInput(netInput)
  end
end

module CommonClusteringCode
   # *** This function should not be called before an entire batch has been processed by the clusterer ***
  def calcLocalFlockingError
    clusters_center_virtual_or_exact = yield
    distanceToWeightedExamplesCenter = (clusters_center_virtual_or_exact - locationOfExample) * exampleImportanceCorrectionFactor(clusters_center_virtual_or_exact, locationOfExample)
    self.localFlockingError = 1.0 * distanceToWeightedExamplesCenter # TODO weightingOfErrorDueToDistanceFromFlocksCenter(algebraicDistanceToFlocksCenter))  # TODO Should 'membershipInFlock(examplesNetInput)' be included?  # If included, it reduces the importance of examples with small io derivatives  # TODO Should 'membershipInFlock(examplesNetInput)' be included -- This term, if included, reduces the importance of examples with small io derivatives  ## TODO Should 'weightingOfErrorDueToDistanceFromFlocksCenter(algebraicDistanceToFlocksCenter)' be included?
    self.accumulatedAbsoluteFlockingError += localFlockingError.abs
    return localFlockingError
  end

  def exampleImportanceCorrectionFactor(clustersCenter, locationOfExample)
    factor = ioDerivativeFromNetInput(locationOfExample) / ioDerivativeFromNetInput(clustersCenter)
  end
end