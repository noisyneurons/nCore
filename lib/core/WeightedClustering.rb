### VERSION "nCore"
## ../nCore/lib/core/WeightedClustering.rb
# This is an implementation of the fuzzy c-means clustering algorithm
# see Fuzzy C-means clustering write-ups; for example, see:
# http://sites.google.com/site/dataclusteringalgorithms/fuzzy-c-means-clustering-algorithm

require_relative 'Utilities'

# Globals, Constants
INFINITY = 1.0/0

class DynamicClusterer
  protected
  attr_reader :args, :numberOfClusters, :m, :delta, :maxNumberOfClusteringIterations,
              :numExamples, :exampleVectorLength, :floorToPreventOverflow
  public
  attr_reader :clusters

  def initialize(args)
    @args = args
    @floorToPreventOverflow = args[:floorToPreventOverflow] ||= 1.0e-10
    @numberOfClusters = args[:numberOfClusters]
    @m = args[:m]
    @numExamples = args[:numExamples]
    @exampleVectorLength = args[:exampleVectorLength]
    @delta = args[:delta] # clustering is finished if we don't have to move any cluster more than a distance of delta (Euclidian distance measure or?)
    @maxNumberOfClusteringIterations = args[:maxNumberOfClusteringIterations]
    @clusters = Array.new(numberOfClusters) { |clusterNumber| Cluster.new(m, numExamples, exampleVectorLength, clusterNumber) }
  end

  def initializationOfClusterCenters(points)
    @clusters.each { |aCluster| aCluster.calcCenterInVectorSpace(points) }
  end

  def clusterData(points)
    largestEuclidianDistanceMoved = 0.0
    maxNumberOfClusteringIterations.times do |iterationNumber|
      forEachExampleDetermineItsFractionalMembershipInEachCluster(points)
      largestEuclidianDistanceMoved = recenterClusters(points)
      return [clusters, iterationNumber, largestEuclidianDistanceMoved] if (largestEuclidianDistanceMoved < delta) # We are finished when the maximum change in any cluster's center was less that 'delta'
    end
    return [clusters, maxNumberOfClusteringIterations, largestEuclidianDistanceMoved]
  end

  def pointsTargetForIterationInFuzzyClustering(pointNumber)
    theExamplesFractionalMembershipInEachCluster = examplesFractionalMembershipInEachCluster(pointNumber)
    sumOfWeightedDeviationsNecessaryToMoveToClusters = Vector.elements(Array.new(exampleVectorLength) { 0.0 })
    sumOfWeights = 0.0
    theExamplesFractionalMembershipInEachCluster.each_with_index do |examplesFractionalMembershipInCluster, indexToCluster|
      sumOfWeightedDeviationsNecessaryToMoveToClusters += examplesFractionalMembershipInCluster * (clusters[indexToCluster].center - point)
      sumOfWeights += examplesFractionalMembershipInCluster
    end
    aggregatedNecessaryVectorDeviation = sumOfWeightedDeviationsNecessaryToMoveToClusters / sumOfWeights
    targetPoint = aggregatedNecessaryVectorDeviation + point
  end

  def estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster(pointNumber)
    theExamplesFractionalMembershipInEachCluster = examplesFractionalMembershipInEachCluster(pointNumber)
    weightedClusterCentersSum = Vector.elements(Array.new(exampleVectorLength) { 0.0 })
    weightingSum = 0.0
    theExamplesFractionalMembershipInEachCluster.each_with_index do |examplesFractionalMembershipInCluster, indexToCluster|
      weightedClusterCentersSum += examplesFractionalMembershipInCluster * clusters[indexToCluster].center
      weightingSum += examplesFractionalMembershipInCluster
    end
    centerOfWeightedClustersForExample = weightedClusterCentersSum / weightingSum
  end

  ################# For reporting / plotting / measures etc...... ###################################
  def determineClusterAssociatedWithExample(pointNumber)
    clusterWeightingsForExample = examplesFractionalMembershipInEachCluster(pointNumber)
    maxWeight = 0.0
    clusterAssociatedWithExample = nil
    std("clusterWeightingsForExample",clusterWeightingsForExample)
    clusterWeightingsForExample.each_with_index do |weightingGivenExampleForCluster, clusterNumber|

      if (weightingGivenExampleForCluster >= maxWeight)
        maxWeight = weightingGivenExampleForCluster
        clusterAssociatedWithExample = clusters[clusterNumber]
      end
    end
    return clusterAssociatedWithExample
  end

  def withinClusterDispersion(points)
    dispersions = clusters.collect { |cluster| cluster.dispersion(points) }
    return dispersions.mean
  end

  def withinClusterDispersionOfInputs(points)
    dispersions = clusters.collect { |cluster| cluster.dispersionOfInputs(points) }
    return dispersions.mean
  end

  def dispersionOfInputsForDPrimeCalculation(points) # TODO currently assumes a large number of samples!!! TODO
    variances = clusters.collect { |cluster| cluster.varianceOfInputs(points) }
    return Math.sqrt(variances.mean)
  end

  def distanceBetween2ClustersForDimension0
    distance = clusters[1].center[0] - clusters[0].center[0]
  end

  ############################ PRIVATE METHODS BELOW ###########################

  protected

  def examplesFractionalMembershipInEachCluster(pointNumber) # This is just a "recall" routine.. since the calculations have already been done...
    return @clusters.collect { |aCluster| aCluster.membershipWeightForEachExample[pointNumber] }
  end

  def forEachExampleDetermineItsFractionalMembershipInEachCluster(points)
    power = 2.0 / (@m - 1.0)
    points.each_with_index do |thePoint, indexToPoint|
      determineThePointsFractionalMembershipInEachCluster(thePoint, indexToPoint, power)
    end
  end

  def determineThePointsFractionalMembershipInEachCluster(thePoint, indexToPoint, power)
    clusters.each do |aCluster|
      membershipForThisPointForThisCluster = calcThisPointsFractionalMembershipToThisCluster(thePoint, aCluster, power)
      aCluster.membershipWeightForEachExample[indexToPoint] = membershipForThisPointForThisCluster
    end
  end

  def calcThisPointsFractionalMembershipToThisCluster(thePoint, selectedCluster, power) # TODO this code is doubly redundant.  It repeats the same calculations for each cluster.  Will be particularly inefficient for more than 2  clusters.
    distanceToSelectedCluster = selectedCluster.center.dist_to(thePoint)
    sumOfRatios = 0.0
    clusters.each do |otherCluster|
      distanceToOtherCluster = otherCluster.center.dist_to(thePoint)

      #distanceToOtherCluster = unless (distanceToOtherCluster.nan?)
      #                           [distanceToOtherCluster, floorToPreventOverflow].max
      #                         else
      #                           floorToPreventOverflow
      #                         end

      distanceToOtherCluster = [distanceToOtherCluster, floorToPreventOverflow].max
      ratio = distanceToSelectedCluster/distanceToOtherCluster
      ratioToAPower = ratio**power
      sumOfRatios += ratioToAPower
    end
    thisPointsFractionalMembershipToSelectedCluster = 1.0 / sumOfRatios
  end

  ## SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE
  #def calcThisPointsFractionalMembershipToThisCluster(thePoint, arbitraryCluster, power) # TODO this code is doubly redundant.  It repeats the same calculations for each cluster.  Will be particularly inefficient for more than 2  clusters.
  #  case exampleVectorLength
  #    when 1
  #      forVectorLengthEq1(arbitraryCluster, power, thePoint)
  #    else
  #      forVectorLengthGT1(arbitraryCluster, power, thePoint)
  #  end
  #end
  #
  #def forVectorLengthEq1(arbitraryCluster, power, thePoint)
  #
  #  case arbitraryCluster.center[0] >= 0.0
  #    when true # cluster's center is on positive side
  #      arbitraryDistance = arbitraryCluster.center[0] - thePoint[0]
  #      return 1.0 if (arbitraryDistance < 0.0)
  #      return pointsMembershipToArbitraryCluster(arbitraryDistance, power, thePoint)
  #    when false # cluster's center is on negative side
  #      arbitraryDistance = thePoint[0] - arbitraryCluster.center[0]
  #      return 1.0 if (arbitraryDistance < 0.0)
  #      return pointsMembershipToArbitraryCluster(arbitraryDistance, power, thePoint)
  #  end
  #end
  #
  #
  #def forVectorLengthGT1(arbitraryCluster, power, thePoint)
  #  arbitraryDistance = arbitraryCluster.center.dist_to(thePoint)
  #  pointsMembershipToArbitraryCluster(arbitraryDistance, power, thePoint)
  #end
  #
  #def pointsMembershipToArbitraryCluster(arbitraryDistance, power, thePoint)
  #  sumOfRatios = 0.0
  #  clusters.each do |comparisonCluster|
  #    comparisonDistance = comparisonCluster.center.dist_to(thePoint)
  #    comparisonDistance = [comparisonDistance, minDistanceAllowed].max # puts floor on comparison distance to avoid "divide by zero"
  #    ratio = arbitraryDistance/comparisonDistance
  #    ratioToAPower = ratio**power
  #    sumOfRatios += ratioToAPower
  #  end
  #  membershipForThisPointForThisArbitraryCluster = membershipSimplificationFunction(1.0 / sumOfRatios)
  #end

  def recenterClusters(points)
    arrayOfDistancesMoved = clusters.collect { |aCluster| aCluster.recenter!(points) }
    keepCentersSymmetrical if (args[:symmetricalCenters]) # TODO may want to include this in the calculation of largest largestEuclidianDistanceMoved
    # arrayOfDistancesMoved = arrayOfDistancesMoved.delete_if { |number| number.nan? }
    largestEuclidianDistanceMoved = arrayOfDistancesMoved.max
    unless (largestEuclidianDistanceMoved.nil?)
      largestEuclidianDistanceMoved
    else
      floorToPreventOverflow
    end
  end

  def keepCentersSymmetrical
    distanceBetween2ClustersOnNetInputDimension = (clusters[1].center[0] - clusters[0].center[0])
    cluster1IsToTheRightOfCluster0 = (distanceBetween2ClustersOnNetInputDimension >= 0.0)
    symmetricalOffset = distanceBetween2ClustersOnNetInputDimension.abs / 2.0

    case exampleVectorLength

      when 1
        if cluster1IsToTheRightOfCluster0
          clusters[1].center = Vector[symmetricalOffset]
          clusters[0].center = Vector[(-1.0 * symmetricalOffset)]
        else
          clusters[0].center = Vector[symmetricalOffset]
          clusters[1].center = Vector[(-1.0 * symmetricalOffset)]
        end


      when 2
        if cluster1IsToTheRightOfCluster0
          clusters[1].center = Vector[symmetricalOffset, clusters[1].center[1]]
          clusters[0].center = Vector[(-1.0 * symmetricalOffset), clusters[0].center[1]]
        else
          clusters[0].center = Vector[symmetricalOffset, clusters[0].center[1]]
          clusters[1].center = Vector[(-1.0 * symmetricalOffset), clusters[1].center[1]]
        end

      else
        STDERR.puts "error: Example Vector Length incorrectly specified"
    end
  end
end


# Fuzzy Cluster class
class Cluster
  attr_reader :m, :numExamples, :examplesVectorLength, :clusterNumber, :dispersion
  attr_accessor :center, :membershipWeightForEachExample

  def initialize(m, numExamples, examplesVectorLength, clusterNumber=0)
    @m = m
    @numExamples = numExamples
    @examplesVectorLength = examplesVectorLength
    @clusterNumber = clusterNumber
    self.randomlyInitializeExampleMembershipWeights
  end

  def randomlyInitializeExampleMembershipWeights # is this the place for this initialization?
    self.membershipWeightForEachExample = Array.new(numExamples) { rand**m } # { rand }  # TODO is it useful to use **m here?  # TODO Alternative:a you could randomly pick an example to be the initial center for the cluster.  Would do this "above the cluster class."
  end

  # Recenters the cluster
  def recenter!(examples)
    STDERR.puts "Error: Number of Examples is INCORRECT!!" if (numExamples != examples.length)
    old_center = center
    self.calcCenterInVectorSpace(examples)
    return old_center.dist_to(center) # this is currently a Euclidian Distance Measure.
  end

  def calcCenterInVectorSpace(examples)
    sumOfWeightedExamples = sumUpExamplesWeightedByMembershipInThisCluster(examples)
    sumOfWeights = membershipWeightForEachExample.inject { |sum, value| sum + (value**m) }
    self.center = sumOfWeightedExamples / sumOfWeights
  end

  def dispersion(examples) # This actually calculates the standard deviation (unadjusted for small N)
    return nil if (examples.size < 2)
    sumOfWeightedDistancesSquared = 0.0
    sumOfWeights = 0.0
    membershipWeightForEachExample.each_with_index do |theExamplesWeighting, indexToExample|
      adjustedWeight = theExamplesWeighting**m # TODO 'AdjustedWeight' comes from the 'theory' behind fuzzy c-means clustering.  Need to double check my understanding of the appropriateness of this construct 'AdjustedWeight**m'
      distanceBetweenCenterAndExample = (examples[indexToExample] - center).r
      sumOfWeightedDistancesSquared += adjustedWeight * distanceBetweenCenterAndExample**2.0
      sumOfWeights += adjustedWeight
    end
    weightedStandardDeviation = Math.sqrt(sumOfWeightedDistancesSquared / sumOfWeights)
    @dispersion = weightedStandardDeviation
    return @dispersion
  end

  def varianceOfInputs(examples)
    examplesWithJustInputComponent = examples.collect { |anExample| Vector[anExample[0]] }
    return nil if (examplesWithJustInputComponent.size < 2)
    sumOfWeightedDistancesSquared = 0.0
    sumOfWeights = 0.0
    membershipWeightForEachExample.each_with_index do |theExamplesWeighting, indexToExample|
      adjustedWeight = theExamplesWeighting**m
      distanceBetweenCenterAndExample = (examplesWithJustInputComponent[indexToExample] - Vector[@center[0]]).r
      sumOfWeightedDistancesSquared += adjustedWeight * distanceBetweenCenterAndExample**2.0
      sumOfWeights += adjustedWeight
    end
    weightedVariance = sumOfWeightedDistancesSquared / sumOfWeights
    return weightedVariance
  end

  def dispersionOfInputs(examples)
    weightedStandardDeviation = Math.sqrt(varianceOfInputs(examples))
  end

  def <=>(secondCluster)
    self.center.r <=> secondCluster.center.r
  end

  def to_s
    description = "\t\tcenter=\t#{@center}\n"
    membershipWeightForEachExample.each_with_index do |exampleWeight, pointNumber|
      description += "\t\tExample Number: #{pointNumber}\tWeight: #{exampleWeight}"
    end
    return description
  end

  protected

  def sumUpExamplesWeightedByMembershipInThisCluster(examples)
    sumOfWeightedExamples = Vector.elements(Array.new(examplesVectorLength, 0.0), copy=false)
    membershipWeightForEachExample.each_with_index do |anExampleWeight, indexToExample|
      sumOfWeightedExamples += (examples[indexToExample] * anExampleWeight**m)
    end
    return sumOfWeightedExamples
  end
end


#class FuzzyClustererOfExamplesOfDifferingImportance  < DynamicClusterer
#  def initialize(args)
#    @args = args
#    @floorToPreventOverflow = args[:floorToPreventOverflow] ||= 1.0e-10
#    @numberOfClusters = args[:numberOfClusters]
#    @m = args[:m]
#    @numExamples = args[:numExamples]
#    @exampleVectorLength = args[:exampleVectorLength]
#    @delta = args[:delta] # clustering is finished if we don't have to move any cluster more than a distance of delta (Euclidian distance measure or?)
#    @maxNumberOfClusteringIterations = args[:maxNumberOfClusteringIterations]
#    @clusters = Array.new(numberOfClusters) { |clusterNumber| ClusterWithExamplesOfDifferingImportance.new(m, numExamples, exampleVectorLength, clusterNumber) }
#  end
#end


#class ClusterWithExamplesOfDifferingImportance  < Cluster   # TODO still do not understand the small difference in outcome for these two versions of Cluster!
#  include CommonNeuronCalculations
#
#  def calcCenterInVectorSpace(examples)
#    sumOfWeightedExamples = sumUpExamplesWeightedByMembershipInThisCluster(examples)
#    self.center = sumOfWeightedExamples / sumTheWeightsTimesTheExamplesImportance(examples)
#  end
#
#  def sumUpExamplesWeightedByMembershipInThisCluster(examples)
#    sum = Vector.elements(Array.new(examplesVectorLength, 0.0), copy=false)
#    membershipWeightForEachExample.each_with_index do |anExampleWeight, indexToExample|
#      example = examples[indexToExample]
#      sum += ( examplesImportance(example) * anExampleWeight**m  * example )
#    end
#    return sum
#  end
#
#  def sumTheWeightsTimesTheExamplesImportance(examples)
#    sum = 0.0
#    membershipWeightForEachExample.each_with_index do |anExampleWeight, indexToExample|
#      example = examples[indexToExample]
#      sum += ( examplesImportance(example) * anExampleWeight**m )
#    end
#    return sum
#  end
#
#  def examplesImportance(example)
#    netInput = example[0]
#    ioDerivativeFromNetInput(netInput)
#  end
#end
#
