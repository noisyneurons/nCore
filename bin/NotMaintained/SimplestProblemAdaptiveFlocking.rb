### VERSION "nCore"
## ../nCore/bin/SimplestProblemAdaptiveFlocking.rb
# Purpose:  To quantitatively explore the simplest clustering w/o supervision.
# This is a simplified and significantly reorganized version of 'Phase1Phase2MultiCycle.rb'

require 'yaml'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralPartsExtended'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers.rb'

module CommonNeuronCalculations
  public
  def clearWithinEpochMeasures
    metricRecorder.clearWithinEpochMeasures
    upperErAry.clear
    flockErAry.clear
  end
end

class FlockingOutputNeuron < OutputNeuron
  attr_accessor :upperErAry, :flockErAry, :flockErrorMultiplier

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
    @clusterer = typeOfClusterer.new(args)
    @store = {}
    @dPrime = 0.0
    @trainingSequence = TrainingSequence.instance
    @upperErAry = []
    @flockErAry = []
    @flockErrorMultiplier = 0.0
  end
end

class SimpleFlockingNeuronNetwork
  def designateAndLabelGroupsOfNeurons
    @allNeuronsInOneArray = allNeuronLayers.flatten
    @inputLayer = allNeuronLayers[0]
    @hiddenLayer = nil
    @outputLayer = allNeuronLayers[1]
    @neuronsWithInputLinks = outputLayer
    @neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
  end
end

class Trainer
  def designateNeuronGroups
    neuronsCreatingFlockingError = outputLayer
    neuronsAdaptingToLocalFlockingError = outputLayer
    neuronsAdaptingToBackPropedFlockingError = []
    return neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError
  end

  ####------------ Adaption to both Output and Flocking Error ------------------------------

  def measureAndStoreAllNeuralResponsesToBoth(learningRate, neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsAdaptingToLocalFlockingError.each do |aNeuron|
      if (aNeuron.flockErAry.length > 0)

        magnitudeOfUpperError = (Vector.elements(aNeuron.upperErAry)).magnitude
        magnitudeOfFlockError = (Vector.elements(aNeuron.flockErAry)).magnitude
        aNeuron.flockErrorMultiplier = 2.0 * (magnitudeOfUpperError / magnitudeOfFlockError)

        #maxUpperError = aNeuron.upperErAry.max
        #maxFlockError = aNeuron.flockErAry.max
        #aNeuron.flockErrorMultiplier = 1.0 * (maxUpperError / maxFlockError)
      else
        aNeuron.flockErrorMultiplier = 0.0
      end
    end
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    neuronsWithInputLinks.each { |neuron| neuron.learningRate = learningRate }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }

      neuronsCreatingFlockingError.each do |aNeuron|
        dataRecorded = aNeuron.recordResponsesForExample
        localFlockingError = aNeuron.calcLocalFlockingError
        dataRecorded[:localFlockingError] = localFlockingError
        aNeuron.recordResponsesForExampleToDB(dataRecorded)
      end

      # neuronsAdaptingToLocalFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |errorFromUpperLayers, localFlockError| errorFromUpperLayers - localFlockError } }
      neuronsAdaptingToLocalFlockingError.each do |aNeuron|
        aNeuron.calcDeltaWsAndAccumulate do |errorFromUpperLayers, localFlockError|
          aNeuron.upperErAry << errorFromUpperLayers.abs
          aNeuron.flockErAry << localFlockError.abs
          #puts "upperError, FlockError=\t#{errorFromUpperLayers}\t#{localFlockError}"
          (errorFromUpperLayers - (aNeuron.flockErrorMultiplier * localFlockError))
        end
      end
    end
  end
end

################################ Mods for reporting
class FlockingNeuronRecorder ##  TODO temporary
  def recordResponsesForEpoch
    if (trainingSequence.timeToRecordData)
      determineCentersOfClusters()
      epochDataToRecord = ({:epochNumber => dataStoreManager.epochNumber, :neuronID => neuron.id,
                            :wt1 => neuron.inputLinks[0].weight, :wt2 => neuron.inputLinks[1].weight,
                            :cluster0Center => @cluster0Center, :cluster1Center => @cluster1Center,
                            :dPrime => neuron.dPrime})
      quickReportOfExampleWeightings(epochDataToRecord)
      epochDataSet.insert(epochDataToRecord)
    end
  end

  def quickReportOfExampleWeightings(epochDataToRecord)
    neuron.clusterer.clusters.each_with_index do |cluster, numberOfCluster|
      cluster.exampleMembershipWeightsForCluster.each { |exampleWeight| puts "Epoch Number, Cluster Number and Example Weighting= #{epochDataToRecord[:epochNumber]}\t#{numberOfCluster}\t#{exampleWeight}" }
      puts
      puts "NumExamples=\t#{cluster.numExamples}\tNum Membership Weights=\t#{cluster.exampleMembershipWeightsForCluster.length}"
    end
  end
end
####################### Mods for reporting

def createTrainingSet(args)
  include ExampleDistribution
  examples = []
  examples << {:inputs => [1.0, 1.0], :targets => [1.0], :exampleNumber => 0, :class => 1}
  examples << {:inputs => [1.0, 2.0], :targets => [1.0], :exampleNumber => 1, :class => 1}
  examples << {:inputs => [1.0, 3.0], :targets => [1.0], :exampleNumber => 2, :class => 1}
  examples << {:inputs => [1.0, 4.0], :targets => [1.0], :exampleNumber => 3, :class => 1}
  examples << {:inputs => [-1.0, -1.0], :targets => [0.0], :exampleNumber => 4, :class => 0}
  examples << {:inputs => [-1.0, -2.0], :targets => [0.0], :exampleNumber => 5, :class => 0}
  examples << {:inputs => [-1.0, -3.0], :targets => [0.0], :exampleNumber => 6, :class => 0}
  examples << {:inputs => [-1.0, -4.0], :targets => [0.0], :exampleNumber => 7, :class => 0}
  examples
end

def displayAndPlotResults(args, dPrimes, dataStoreManager, lastEpoch,
    lastTestingMSE, lastTrainingMSE, network, theTrainer, trainingSequence)
  puts network
  puts "Elapsed Time=\t#{theTrainer.elapsedTime}"
  puts "\tAt Epoch #{trainingSequence.epochs}"
  puts "\tAt Epoch #{lastEpoch}"
  puts "\t\tThe Network's Training MSE=\t#{lastTrainingMSE}\t and TEST MSE=\t#{lastTestingMSE}\n"
  puts "\t\t\tThe dPrime(s) at the end of training are: #{dPrimes}"

#############################  plotting and visualization....
  plotMSEvsEpochNumber(network)

  dataSetFromJoin = dataStoreManager.joinDataSets # joinForShoesDisplay
  dataStoreManager.transferDataSetToVisualizer(dataSetFromJoin, args)
end

def setParameters
  # You can greatly speed things up (but with a penalty of poorer flocking) by increasing
  # bpLearningRate and flockLearningRate by 100.  Then 'correcting' flockLearningRate by
  # reducing it by 5-10

  numberOfExamples = 8
  args = {
      # parameters that impact learning dynamics:
      :learningRate => 1.0,
      :phase1Epochs => 50, #
      :phase2Epochs => 0, # 1000, # 100,

      # Stop training parameters
      :minMSE => 0.001,
      :maxNumEpochs => 4e3,
      :leadingFactor => 1.0, # 1.02, #   1.0

      # Network Architecture
      :numberOfInputNeurons => 2,
      :numberOfHiddenNeurons => 0,
      :numberOfOutputNeurons => 1,
      :weightRange => 1.0,

      # Training Set parameters
      :numberOfExamples => numberOfExamples,

      # Recording and database parameters
      :numberOfEpochsBetweenStoringDBRecords => 1,

      # Flocking Parameters...
      :typeOfClusterer => DynamicClusterer,
      :numberOfClusters => 2,
      :m => 2.0,
      :numExamples => numberOfExamples,
      :exampleVectorLength => 2,
      :delta => 1e-3,
      :maxNumberOfClusteringIterations => 100,
      :symmetricalCenters => false, # if true, speed is negatively affected

      # Inner Numeric Constraints
      :minDistanceAllowed => 1e-30
  }
end

###################################### Start of Main ##########################################
srand(0)
args = setParameters()

############################### create training set...
examples = createTrainingSet(args)

######################## Specify data store and experiment description....
descriptionOfExperiment = "SimplestProblemAdaptiveFlocking  -- Just Created"
databaseFilename = "acrossEpochsSequel" #  = ""
dataStoreManager = SimulationDataStoreManager.create(databaseFilename, examples, args, descriptionOfExperiment)

######################## Create Network....
network = SimpleFlockingNeuronNetwork.new(dataStoreManager, args)

############################### train ...
trainingSequence = TrainingSequence.create(network, args)
theTrainer = Trainer.new(trainingSequence, network, args)
lastEpoch, lastTrainingMSE, dPrimes = theTrainer.simpleLearningWithFlocking(examples)

lastTestingMSE = nil
dPrimes = [nil]
theTrainer.storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, dPrimes)

displayAndPlotResults(args, dPrimes, dataStoreManager, lastEpoch, lastTestingMSE,
                      lastTrainingMSE, network, theTrainer, trainingSequence)

