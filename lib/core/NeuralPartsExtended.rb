### VERSION "nCore"
## ../nCore/lib/core/NeuralPartsExtended.rb

require_relative 'Utilities'
require_relative 'NeuralParts'
require_relative 'WeightedClustering'

############################################################
module CommonNeuronCalculations
  def calcAccumDeltaWsForHigherLayerError
    inputLinks.each { |inputLink| inputLink.calcAccumDeltaWsForHigherLayerError(higherLayerError) }
  end

  def recordResponsesForEpoch
    metricRecorder.recordResponsesForEpoch
  end

  def recordLocalFlockingError
    metricRecorder.recordLocalFlockingError
  end
end

module CommonClusteringCode
  attr_accessor :netInput, :higherLayerError, :errorToBackPropToLowerLayer,
                :clusterer, :clusters, :localFlockingError, :accumulatedAbsoluteFlockingError,
                :maxNumberOfClusteringIterations, :dPrime, :flockingTargeter, :targetToFlockTo

  def examplesContainedInEachCluster
    clusters.collect { |aCluster| examplesInCluster(aCluster) }
  end

  def examplesInCluster(aCluster)
    aCluster.dominantExamplesForCluster
  end

  def initializeClusterCenters
    arrayOfVectorsRepresentingPointsInSpace = metricRecorder.vectorizeEpochMeasures
    clusterer.initializationOfClusterCenters(arrayOfVectorsRepresentingPointsInSpace)
  end

  def clusterAllResponses
    arrayOfVectorsRepresentingPointsInSpace = metricRecorder.vectorizeEpochMeasures
    dummy, iterationNumber, largestEuclidianDistanceMoved = clusterer.clusterData(arrayOfVectorsRepresentingPointsInSpace)
    STDERR.puts "exceeded max number of clustering iterations.  Maximum number=  #{maxNumberOfClusteringIterations}" if (iterationNumber > maxNumberOfClusteringIterations)
    STDERR.puts "too big a 'move'. The move was=  #{largestEuclidianDistanceMoved}" if (largestEuclidianDistanceMoved > 0.01)
    return iterationNumber
  end

  # *** This function should not be called before an entire batch has been processed by the clusterer ***
  def calcLocalFlockingError
    keepTargetsSymmetrical if (args[:keepTargetsSymmetrical])
    errors = clusters.collect do |aCluster|
      distanceBetweenClusterAndExample = aCluster.center[0] - locationOfExample
      weightedDistance = functionForWeightingDistanceBetweenExampleAndClustersCenter(distanceBetweenClusterAndExample)
      errorTakingExamplesMembershipIntoAccount = weightedDistance * aCluster.membershipWeightForEachExample[exampleNumber]
    end
    self.localFlockingError = errors.reduce(:+)
    self.accumulatedAbsoluteFlockingError += localFlockingError.abs
    return localFlockingError
  end

  def functionForWeightingDistanceBetweenExampleAndClustersCenter(distanceBetweenClusterAndExample)
    # distanceBetweenClusterCenters = (clusters[1].center - clusters[0].center).abs      # TODO use this function?
    return distanceBetweenClusterAndExample
  end

  def keepTargetsSymmetrical
    distanceBetween2TargetsOnNetInputDimension = (clusters[1].center[0] - clusters[0].center[0])
    target1IsToTheRightOfTarget0 = (distanceBetween2TargetsOnNetInputDimension >= 0.0)
    symmetricalOffset = distanceBetween2TargetsOnNetInputDimension.abs / 2.0

    if target1IsToTheRightOfTarget0
      clusters[1].center = Vector[symmetricalOffset]
      clusters[0].center = Vector[(-1.0 * symmetricalOffset)]
    else
      clusters[0].center = Vector[symmetricalOffset]
      clusters[1].center = Vector[(-1.0 * symmetricalOffset)]
    end
  end

  def calcAccumDeltaWsForLocalFlocking
    inputLinks.each { |inputLink| inputLink.calcAccumDeltaWsForLocalFlocking(localFlockingError) }
  end

  ## ---------------- Reporting methods ------------------------------

  def calc_dPrime(arrayOfVectorsRepresentingPointsInSpace) # very inexact for small numbers of samples... need to use F-st...
                                                           #puts "dispersionOfInputsForDPrimeCalculation=\t#{clusterer.dispersionOfInputsForDPrimeCalculation(arrayOfVectorsRepresentingPointsInSpace)}"
                                                           #puts "distanceBetween2ClustersForDimension0=\t#{clusterer.distanceBetween2ClustersForDimension0}"
    return (clusterer.distanceBetween2ClustersForDimension0 / clusterer.dispersionOfInputsForDPrimeCalculation(arrayOfVectorsRepresentingPointsInSpace)).abs
  end

  def calc_dispersion(arrayOfVectorsRepresentingPointsInSpace) # very inexact for small numbers of samples... need to use F-st...
    return clusterer.dispersionOfInputsForDPrimeCalculation(arrayOfVectorsRepresentingPointsInSpace)
  end

  private ############################ PRIVATE METHODS BELOW ###########################

  def locationOfExample # TODO Here we have EXCLUDED other dimensions
    return netInput
  end

## Currently Dormant Code

  def centerOfDominantClusterForExample
    dominantCluster = clusterer.determineClusterAssociatedWithExample(exampleNumber)
    return dominantCluster.center[0]
  end
end

############################################################

class FlockingNeuron < Neuron
  include CommonClusteringCode

  def postInitialize
    @inputLinks = []
    @netInput = 0.0
    self.output = self.ioFunction(@netInput) # Only doing this in case we wish to use this code for recurrent networks
    @outputLinks = []
    @higherLayerError = 0.0
    @errorToBackPropToLowerLayer = 0.0
    @localFlockingError = 0.0
    @metricRecorder= FlockingNeuronRecorder.new(self, args)
    @exampleNumber = nil
    @maxNumberOfClusteringIterations = args[:maxNumberOfClusteringIterations]
    typeOfClusterer = args[:typeOfClusterer]
    @clusterer = typeOfClusterer.new(args)
    @clusters = @clusterer.clusters
    @dPrime = 0.0
    @trainingSequence = args[:trainingSequence]
  end

  def backPropagate(&block)
    self.higherLayerError = calcNetError * ioDerivativeFromOutput(output)
    self.errorToBackPropToLowerLayer = higherLayerError
    self.errorToBackPropToLowerLayer = yield(higherLayerError, localFlockingError) if (block.present?)
  end
end

class FlockingOutputNeuron < OutputNeuron
  include CommonClusteringCode

  def postInitialize
    self.output = ioFunction(@netInput = 0.0) # Only doing this in case we wish to use this code for recurrent networks
    @inputLinks = []
    @netInput = 0.0
    @higherLayerError = 0.0
    @errorToBackPropToLowerLayer = 0.0
    @localFlockingError = 0.0
    @arrayOfSelectedData = nil
    @keyToExampleData = :targets
    @exampleNumber = nil
    @metricRecorder= FlockingOutputNeuronRecorder.new(self, args)
    @maxNumberOfClusteringIterations = args[:maxNumberOfClusteringIterations]
    typeOfClusterer = args[:typeOfClusterer]
    @clusterer = typeOfClusterer.new(args)
    @clusters = @clusterer.clusters
    @dPrime = 0.0
    @trainingSequence = args[:trainingSequence]
  end

  def backPropagate(&block)
    self.higherLayerError = outputError * ioDerivativeFromOutput(output)
    self.errorToBackPropToLowerLayer = higherLayerError
    self.errorToBackPropToLowerLayer = yield(higherLayerError, localFlockingError) if (block.present?)
  end
end

############################################################

class FlockingLink < Link
  attr_accessor :previousDeltaWAccumulated, :store

  def calcAccumDeltaWsForHigherLayerError(higherLayerError)
    self.deltaW = learningRate * higherLayerError * inputNeuron.output
    self.deltaWAccumulated += deltaW
  end

  def calcAccumDeltaWsForLocalFlocking(localFlockingError)
    calcAccumDeltaW(localFlockingError, alpha= 0.7)
  end

  def calcAccumDeltaW(anError, alpha)
    self.deltaW = learningRate * anError * inputNeuron.output
    augDeltaW = augmentChangeUsingMomentum(deltaW, alpha)
    self.deltaWAccumulated += augDeltaW
  end

  def augmentChangeUsingMomentum(change, alpha)
    oldAugmentedChange = store
    augmentedChange = change + (alpha * oldAugmentedChange)
    self.store = augmentedChange
  end

  def backPropagate
    return outputNeuron.errorToBackPropToLowerLayer * weight
  end

  def addAccumulationToWeight
    self.weight = weight - deltaWAccumulated
  end

  def calcDeltaW
    STDERR.puts " ERROR from a FlockingLink: The method 'calcDeltaW' was called!"
  end
end

############################################################

class FlockingNeuronRecorder < NeuronRecorder # TODO Need to separate into 2 classes the two concerns currently handled by this class: reporting vs. getting info for 'clusterAllResponses'
  attr_accessor :trainingSequence, :exampleDataSet, :epochDataSet, :dataStoreManager, :exampleVectorLength

  def initialize(neuron, args)
    super(neuron, args)
    @exampleVectorLength = args[:exampleVectorLength]
  end

  def dataToRecord
    aHash = super
    return aHash.merge({:higherLayerError => neuron.higherLayerError,
                        :errorToBackPropToLowerLayer => neuron.errorToBackPropToLowerLayer,
                        :localFlockingError => neuron.localFlockingError,
                        :targetToFlockTo => neuron.targetToFlockTo})
  end

  def recordLocalFlockingError
    withinEpochMeasures.last[:localFlockingError] = neuron.localFlockingError
  end

  def vectorizeEpochMeasures
    convertEachHashToAVector(withinEpochMeasures)
  end

  private

  def convertEachHashToAVector(anArray)
    return anArrayOfVectors = anArray.collect do |measuresForAnExample|
      netInputDistance = measuresForAnExample[:netInput]
      case exampleVectorLength
        when 1
          Vector[netInputDistance]
        when 2
          Vector[netInputDistance, (measuresForAnExample[:higherLayerError])]
      end
    end
  end

  #  SAVE SAVE SAVE  SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE  SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE SAVE
  #def recordResponsesForEpoch
  #  if (trainingSequence.timeToRecordData)
  #    determineCentersOfClusters()
  #    epochDataToRecord = ({:epochNumber => dataStoreManager.epochNumber, :neuronID => neuron.id,
  #                          :wt1 => neuron.inputLinks[0].weight, :wt2 => neuron.inputLinks[1].weight,
  #                          :cluster0Center => @cluster0Center, :cluster1Center => @cluster1Center,
  #                          :dPrime => neuron.dPrime})
  #    quickReportOfExampleWeightings(epochDataToRecord)
  #    NeuronData.new(epochDataToRecord)
  #  end
  #end
  #
  #def quickReportOfExampleWeightings(epochDataToRecord)
  #  neuron.clusters.each_with_index do |cluster, numberOfCluster|
  #    cluster.membershipWeightForEachExample.each { |exampleWeight| puts "Epoch Number, Cluster Number and Example Weighting= #{epochDataToRecord[:epochNumber]}\t#{numberOfCluster}\t#{exampleWeight}" }
  #    puts
  #    puts "NumExamples=\t#{cluster.numExamples}\tNum Membership Weights=\t#{cluster.membershipWeightForEachExample.length}"
  #  end
  #end
  #
  #def determineCentersOfClusters
  #  cluster0 = neuron.clusters[0]
  #  if (cluster0.center.present?)
  #    @cluster0Center = cluster0.center[0]
  #    cluster1 = neuron.clusters[1]
  #    @cluster1Center = cluster1.center[0]
  #  else
  #    cluster0Center = 0.0
  #    cluster1Center = 0.0
  #  end
  #end
end

class FlockingOutputNeuronRecorder < FlockingNeuronRecorder # TODO Need to separate into 2 classes the two concerns currently handled by this class: reporting vs. getting info for 'clusterAllResponses'
  def dataToRecord
    aHash = super
    aHash[:weightedErrorMetric] = neuron.weightedErrorMetric
    return aHash
  end
end
