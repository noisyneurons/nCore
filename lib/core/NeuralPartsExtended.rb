### VERSION "nCore"
## ../nCore/lib/core/NeuralPartsExtended.rb

require_relative 'Utilities'
require_relative 'NeuralParts'
require_relative 'WeightedClustering'


############################################################
module CommonNeuronCalculations
  def recordResponsesForEpoch
    metricRecorder.recordResponsesForEpoch
  end

  def recordLocalFlockingError
    metricRecorder.recordLocalFlockingError
  end

  def recordResponsesForExampleToDB(data)
    metricRecorder.recordResponsesForExampleToDB(data)
  end
end

module CombiningFlockingAndSupervisedErrorCode

  def calcAccumDeltaWsForOutputError
    inputLinks.each { |inputLink| inputLink.calcAccumDeltaWsForOutputError(higherLayerError) }
  end

  def calcAccumDeltaWsForLocalFlocking
    inputLinks.each { |inputLink| inputLink.calcAccumDeltaWsForLocalFlocking(localFlockingError) }
  end
end

module CommonClusteringCode
#  include DistanceTransforms

  def clusters
    clusterer.clusters
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
    clusters_center_virtual_or_exact = yield
    distanceToWeightedExamplesCenter = (clusters_center_virtual_or_exact - locationOfExample)
    self.localFlockingError = 1.0 * distanceToWeightedExamplesCenter # TODO weightingOfErrorDueToDistanceFromFlocksCenter(algebraicDistanceToFlocksCenter))  # TODO Should 'membershipInFlock(examplesNetInput)' be included?  # If included, it reduces the importance of examples with small io derivatives  # TODO Should 'membershipInFlock(examplesNetInput)' be included -- This term, if included, reduces the importance of examples with small io derivatives  ## TODO Should 'weightingOfErrorDueToDistanceFromFlocksCenter(algebraicDistanceToFlocksCenter)' be included?
    self.accumulatedAbsoluteFlockingError += localFlockingError.abs
    return localFlockingError
  end

  def weightedExamplesCenter # TODO !! should only need to this on the first flocking iteration for each example ('memoize' this??)
    clusterer.estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster(exampleNumber)[0]
  end

  def centerOfDominantClusterForExample
    dominantCluster = clusterer.determineClusterAssociatedWithExample(exampleNumber)
    return dominantCluster.center[0]
  end

  ## ---------------- Reporting methods ------------------------------

  def reportFlockingInformation
    membershipForExampleInCluster0 =(clusterer.clusters[0]).membershipWeightForEachExample[exampleNumber]
    membershipForExampleInCluster1 =(clusterer.clusters[1]).membershipWeightForEachExample[exampleNumber]
    dataForReport = {:neuronID => self.id,
                     :localFlockingError => self.localFlockingError,
                     :weightedExamplesCenter => weightedExamplesCenter,
                     :locationOfExample => locationOfExample,
                     :membershipForExampleInCluster0 => membershipForExampleInCluster0,
                     :membershipForExampleInCluster1 => membershipForExampleInCluster1
    }
    return dataForReport
  end

  def calc_dPrime(arrayOfVectorsRepresentingPointsInSpace) # very inexact for small numbers of samples... need to use F-st...
                                                           #puts "dispersionOfInputsForDPrimeCalculation=\t#{clusterer.dispersionOfInputsForDPrimeCalculation(arrayOfVectorsRepresentingPointsInSpace)}"
                                                           #puts "distanceBetween2ClustersForDimension0=\t#{clusterer.distanceBetween2ClustersForDimension0}"
    return (clusterer.distanceBetween2ClustersForDimension0 / clusterer.dispersionOfInputsForDPrimeCalculation(arrayOfVectorsRepresentingPointsInSpace)).abs
  end

  def calc_dispersion(arrayOfVectorsRepresentingPointsInSpace) # very inexact for small numbers of samples... need to use F-st...
    return clusterer.dispersionOfInputsForDPrimeCalculation(arrayOfVectorsRepresentingPointsInSpace)
  end


  private ############################ PRIVATE METHODS BELOW ###########################

  def locationOfExample
    return netInput
  end
end

############################################################

class FlockingNeuron < Neuron
  attr_accessor :localFlockingError, :accumulatedAbsoluteFlockingError,
                :higherLayerError, :errorToBackPropToLowerLayer,
                :clusterer, :dPrime, :trainingSequence,
                :flockingGain, :layerGain, :maxNumberOfClusteringIterations
  include CombiningFlockingAndSupervisedErrorCode
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
    @dPrime = 0.0
    @trainingSequence = TrainingSequence.instance
  end

  def backPropagate(&block)
    self.higherLayerError = calcNetError * ioDerivativeFromOutput(output)
    self.errorToBackPropToLowerLayer = higherLayerError
    self.errorToBackPropToLowerLayer = yield(higherLayerError, localFlockingError) if (block.present?)
  end
end

class FlockingOutputNeuron < OutputNeuron
  attr_accessor :netInput, :localFlockingError, :accumulatedAbsoluteFlockingError,
                :higherLayerError, :errorToBackPropToLowerLayer, :clusterer,
                :dPrime, :trainingSequence, :flockingGain, :layerGain, :maxNumberOfClusteringIterations
  include CombiningFlockingAndSupervisedErrorCode
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
    typeOfClusterer = args[:typeOfClusterer]
    @maxNumberOfClusteringIterations = args[:maxNumberOfClusteringIterations]
    @clusterer = typeOfClusterer.new(args)
    @dPrime = 0.0
    @trainingSequence = TrainingSequence.instance
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

  def calcAccumDeltaWsForOutputError(higherLayerError)
    self.store = 0.0
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
    STDERR.puts " ERROR Link.calcDeltaW called"
    # self.deltaW = learningRate * outputNeuron.error * inputNeuron.output
  end
end

############################################################

class NeuronRecorder
  attr_accessor :trainingSequence, :exampleDataSet, :epochDataSet, :dataStoreManager

  def initialize(neuron, args)
    @neuron = neuron
    @args = args
    @trainingSequence = TrainingSequence.instance
    @dataStoreManager = SimulationDataStoreManager.instance
    @exampleDataSet = dataStoreManager.exampleDataSet
    @epochDataSet = dataStoreManager.epochDataSet
    @withinEpochMeasures = []
  end

  def recordResponsesForExampleToDB(exampleDataToRecord)
    exampleDataSet.insert(exampleDataToRecord) if (trainingSequence.timeToRecordData)
  end
end

class FlockingNeuronRecorder < NeuronRecorder # TODO Need to separate into 2 classes the two concerns currently handled by this class: reporting vs. getting info for 'clusterAllResponses'
  attr_accessor :trainingSequence, :exampleDataSet, :epochDataSet, :dataStoreManager, :exampleVectorLength

  def initialize(neuron, args)
    @neuron = neuron
    @args = args
    @trainingSequence = TrainingSequence.instance
    @dataStoreManager = SimulationDataStoreManager.instance
    @exampleDataSet = dataStoreManager.exampleDataSet
    @epochDataSet = dataStoreManager.epochDataSet
    @withinEpochMeasures = []
    @exampleVectorLength = args[:exampleVectorLength]
  end

  def recordResponsesForExample
    exampleDataToRecord = ({:epochNumber => dataStoreManager.epochNumber, :neuronID => neuron.id,
                            :exampleNumber => neuron.exampleNumber, :netInput => neuron.netInput,
                            :higherLayerError => neuron.higherLayerError,
                            :errorToBackPropToLowerLayer => neuron.errorToBackPropToLowerLayer,
                            :localFlockingError => neuron.localFlockingError})
    withinEpochMeasures << exampleDataToRecord
    return exampleDataToRecord
  end

  def recordLocalFlockingError
    withinEpochMeasures.last[:localFlockingError] = neuron.localFlockingError
  end

  def recordResponsesForEpoch
    if (trainingSequence.timeToRecordData)
      determineCentersOfClusters()
      epochDataToRecord = ({:epochNumber => dataStoreManager.epochNumber, :neuronID => neuron.id,
                            :wt1 => neuron.inputLinks[0].weight, :wt2 => neuron.inputLinks[1].weight,
                            :cluster0Center => @cluster0Center, :cluster1Center => @cluster1Center,
                            :dPrime => neuron.dPrime})
      # epochDataSet.insert(epochDataToRecord)
    end
  end

  def determineCentersOfClusters
    cluster0 = neuron.clusters[0]
    if (cluster0.center.present?)
      @cluster0Center = cluster0.center[0]
      cluster1 = neuron.clusters[1]
      @cluster1Center = cluster1.center[0]
    else
      cluster0Center = 0.0
      cluster1Center = 0.0
    end
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
end

class FlockingOutputNeuronRecorder < FlockingNeuronRecorder # TODO Need to separate into 2 classes the two concerns currently handled by this class: reporting vs. getting info for 'clusterAllResponses'
  def recordResponsesForExample
    exampleDataToRecord = ({:epochNumber => dataStoreManager.epochNumber, :neuronID => neuron.id,
                            :exampleNumber => neuron.exampleNumber, :netInput => neuron.netInput,
                            :higherLayerError => neuron.higherLayerError,
                            :errorToBackPropToLowerLayer => neuron.errorToBackPropToLowerLayer,
                            :localFlockingError => neuron.localFlockingError,
                            :weightedErrorMetric => neuron.weightedErrorMetric})
    withinEpochMeasures << exampleDataToRecord
    return exampleDataToRecord
  end
end
