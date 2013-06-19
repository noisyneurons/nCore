### VERSION "nCore"
## ../nCore/bin/LearningRateExperiments.rb
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
      # epochDataSet.insert(epochDataToRecord)
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

def setParameters(descriptionOfExperiment)
  # You can greatly speed things up (but with a penalty of poorer flocking) by increasing
  # bpLearningRate and flockLearningRate by 100.  Then 'correcting' flockLearningRate by
  # reducing it by 5-10

  numberOfExamples = 8
  randomNumberSeed = 0
  args = {
      :descriptionOfExperiment => descriptionOfExperiment,
      :rng => Random.new(randomNumberSeed),

      :maxHistory => 8,
      :balanceOfdPrimeVsDispersion => 0.0, # a value of 1.0 indicates that dPrime is
      # to be the sole metric. a value of 0.0 indicates Dispersion is the sole metric
      :multiplyToEmphasizeFlocking => 2.0, # if value = 0.0 only output error
      # is used to determine weight changes.  If value >> 1.0, then flocking error will
      # be dominant in prescribing weight changes.
      :searchRangeRatio => 2.0,

      :phase1Epochs => 100,
      :phase2Epochs => 0,

      # Stop training parameters
      :minMSE => 0.001,
      :maxNumEpochs => 4e3,

      # Network Architecture
      :numberOfInputNeurons => 2,
      :numberOfHiddenNeurons => 0,
      :numberOfOutputNeurons => 1,
      :weightRange => 1.0,
      :typeOfLink => FlockingLink,

      # Training Set parameters
      :numberOfExamples => numberOfExamples,

      # Recording and database parameters
      :numberOfEpochsBetweenStoringDBRecords => 100,

      # Flocking Parameters...
      :typeOfClusterer => DynamicClusterer,
      :numberOfClusters => 2,
      :m => 2.0,
      :numExamples => numberOfExamples,
      :exampleVectorLength => 2,
      :delta => 1e-3,
      :maxNumberOfClusteringIterations => 100,
      :symmetricalCenters => true, # if true, speed is negatively affected
      :leadingFactor => 1.0, # 1.02, #   1.0

      # Inner Numeric Constraints
      :minDistanceAllowed => 1e-30
  }
end

###################################### Start of Main ##########################################
srand(0)
descriptionOfExperiment = "SimpleAdjustableLearningRateTrainerMultiFlockIterations Reference Run"
args = setParameters(descriptionOfExperiment)

############################### create training set...
examples = createTrainingSet(args)

######################## Specify data store and experiment description....
databaseFilename = "acrossEpochsSequel" #  = ""
dataStoreManager = SimulationDataStoreManager.create(databaseFilename, examples, args)

######################## Create Network....
network = SimpleFlockingNeuronNetwork.new(dataStoreManager, args)
puts network

############################### train ...
trainingSequence = TrainingSequence.create(network, args)
theTrainer = SimpleAdjustableLearningRateTrainer.new(trainingSequence, network, args)

arrayOfNeuronsForIOPlots = nil
lastEpoch, lastTrainingMSE, dispersions = theTrainer.simpleLearningWithFlocking(examples, arrayOfNeuronsForIOPlots)

lastTestingMSE = nil
theTrainer.storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, dispersions)

displayAndPlotResults(args, dispersions, dataStoreManager, lastEpoch, lastTestingMSE,
                      lastTrainingMSE, network, theTrainer, trainingSequence)

