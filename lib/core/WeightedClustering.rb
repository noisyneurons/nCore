### VERSION "nCore"
## ../nCore/lib/core/WeightedClustering.rb
# This is an implementation of the fuzzy c-means clustering algorithm
# see Fuzzy C-means clustering write-ups; for example, see:
# http://sites.google.com/site/dataclusteringalgorithms/fuzzy-c-means-clustering-algorithm

require_relative 'Utilities'

# Globals, Constants
INFINITY = 1.0/0

class Targeter

  attr_accessor :neuron, :clusterer, :clusters, :args, :targetsForFlocking, :exampleVectorLength

  def initialize(neuron, clusterer, args)
    @neuron = neuron
    @clusterer = clusterer
    @clusters = clusterer.clusters
    @args = args
    @targetsForFlocking = []
    @exampleVectorLength = args[:exampleVectorLength]
  end

  def examplesTargetForFlocking(pointNumber)
    theExamplesFractionalMembershipInEachCluster = clusterer.examplesFractionalMembershipInEachCluster(pointNumber)
    weightedTargetsSum = Vector.elements(Array.new(exampleVectorLength) { 0.0 })
    sumOfWeights = 0.0
    clusters.each_with_index { |aCluster, indexToCluster| targetsForFlocking[indexToCluster] = aCluster.center } # initialing targets
    keepTargetsSymmetrical if (args[:keepTargetsSymmetrical])
    targetDivergenceFactor = args[:targetDivergenceFactor]
    targetsForFlocking.collect! { |aTarget| aTarget * targetDivergenceFactor }
    theExamplesFractionalMembershipInEachCluster.each_with_index do |examplesFractionalMembershipInCluster, indexToCluster|
      weightedTargetsSum += examplesFractionalMembershipInCluster * targetsForFlocking[indexToCluster]
      sumOfWeights += examplesFractionalMembershipInCluster
    end
    targetForExample = weightedTargetsSum / sumOfWeights
  end

  def keepTargetsSymmetrical
    distanceBetween2TargetsOnNetInputDimension = (targetsForFlocking[1][0] - targetsForFlocking[0][0])
    target1IsToTheRightOfTarget0 = (distanceBetween2TargetsOnNetInputDimension >= 0.0)
    symmetricalOffset = distanceBetween2TargetsOnNetInputDimension.abs / 2.0

    case exampleVectorLength

      when 1
        if target1IsToTheRightOfTarget0
          targetsForFlocking[1] = Vector[symmetricalOffset]
          targetsForFlocking[0] = Vector[(-1.0 * symmetricalOffset)]
        else
          targetsForFlocking[0] = Vector[symmetricalOffset]
          targetsForFlocking[1] = Vector[(-1.0 * symmetricalOffset)]
        end


      when 2
        if target1IsToTheRightOfTarget0
          targetsForFlocking[1] = Vector[symmetricalOffset, targetsForFlocking[1][1]]
          targetsForFlocking[0] = Vector[(-1.0 * symmetricalOffset), targetsForFlocking[0][1]]
        else
          targetsForFlocking[0] = Vector[symmetricalOffset, targetsForFlocking[0][1]]
          targetsForFlocking[1] = Vector[(-1.0 * symmetricalOffset), targetsForFlocking[1][1]]
        end

      else
        STDERR.puts "error: Example Vector Length incorrectly specified"
    end
  end
end


class DynamicClusterer
  protected
  attr_reader :args, :numberOfClusters, :m, :delta, :maxNumberOfClusteringIterations,
              :numExamples, :exampleVectorLength, :floorToPreventOverflow
  public
  attr_reader :clusters

  def initialize(args)
    @args = args
    @floorToPreventOverflow = args[:floorToPreventOverflow] ||= 1.0e-10
    @min = floorToPreventOverflow
    @max = 1.0 - floorToPreventOverflow
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

  def examplesFractionalMembershipInEachCluster(pointNumber) # This is just a "recall" routine.. since the calculations have already been done...
    return @clusters.collect { |aCluster| aCluster.membershipWeightForEachExample[pointNumber] }
  end

  ################# For reporting / plotting / measures etc...... ###################################
  def determineClusterAssociatedWithExample(pointNumber)
    clusterWeightingsForExample = examplesFractionalMembershipInEachCluster(pointNumber)
    maxWeight = 0.0
    clusterAssociatedWithExample = nil
    std("clusterWeightingsForExample", clusterWeightingsForExample)
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

  # TODO The following 4 routines should be redone to eliminate redundant calculations and to simplify code (e.g., fewer methods!!)

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
      distanceToOtherCluster = bottomClip(distanceToOtherCluster)
      ratio = distanceToSelectedCluster/distanceToOtherCluster
      ratioToAPower = ratio**power
      sumOfRatios += ratioToAPower
    end
    thisPointsFractionalMembershipToSelectedCluster = topAndBottomClip(1.0 / sumOfRatios)
  end

  def recenterClusters(points)
    arrayOfDistancesMoved = clusters.collect { |aCluster| aCluster.recenter!(points) }
    arrayOfDistancesMoved = arrayOfDistancesMoved.delete_if { |number| number.nan? }
    unless (arrayOfDistancesMoved.empty?)
      return arrayOfDistancesMoved.max ||= floorToPreventOverflow
    end
    return floorToPreventOverflow
  end

  def bottomClip(value)
    return @min if (value < @min)
    value
  end

  def topClip(value)
    return @max if (value > @max)
    value
  end

  def topAndBottomClip(value)
    topClip(bottomClip(value))
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
    STDERR.puts "Error: Number of Examples is INCORRECT!! i.e. #{numExamples} versus #{examples.length}" if (numExamples != examples.length)
    old_center = center
    self.calcCenterInVectorSpace(examples)
    return old_center.dist_to(center) # this is currently a Euclidian Distance Measure.
  end

  def calcCenterInVectorSpace(examples)
    sumOfWeightedExamples = sumUpExamplesWeightedByMembershipInThisCluster(examples)
    sumOfWeights = membershipWeightForEachExample.inject { |sum, value| sum + (value**m) }
    self.center = sumOfWeightedExamples / sumOfWeights
  end

  def dominantExamplesForCluster
    (0...numExamples).find_all {|i| membershipWeightForEachExample[i] >= 0.5 }
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
