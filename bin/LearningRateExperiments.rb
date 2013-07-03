### VERSION "nCore"
## ../nCore/bin/LearningRateExperiments.rb
# Purpose:  To quantitatively explore the simplest clustering w/o supervision.
# This is a simplified and significantly reorganized version of 'Phase1Phase2MultiCycle.rb'

require 'yaml'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralPartsExtended'

require_relative '../lib/core/ExampleImportanceMods'

require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers.rb'

require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

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
    neuron.clusters.each_with_index do |cluster, numberOfCluster|
      cluster.membershipWeightForEachExample.each { |exampleWeight| puts "Epoch Number, Cluster Number and Example Weighting= #{epochDataToRecord[:epochNumber]}\t#{numberOfCluster}\t#{exampleWeight}" }
      puts
      puts "NumExamples=\t#{cluster.numExamples}\tNum Membership Weights=\t#{cluster.membershipWeightForEachExample.length}"
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

  numberOfExamples = 8
  randomNumberSeed = 0

  args = {
      :descriptionOfExperiment => descriptionOfExperiment,
      :rng => Random.new(randomNumberSeed),

      :phase1Epochs => 10000,
      :phase2Epochs => 0,

      # training parameters re. Output Error
      :outputErrorLearningRate => 0.02,
      :minMSE => 0.0001,
      :maxNumEpochs => (4e3),

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
      :flockingLearningRate => -0.002,
      :maxFlockingIterationsCount => 2000,
      :maxAbsFlockingErrorsPerExample => 0.002, # 0.005,   # 0.04 / numberOfExamples = 0.005

      :typeOfClusterer => DynamicClusterer,
      :numberOfClusters => 2,
      :m => 2.0,
      :numExamples => numberOfExamples,
      :exampleVectorLength => 1,
      :delta => 1e-2,
      :maxNumberOfClusteringIterations => 10,
      :symmetricalCenters => true, # if true, speed is negatively affected

      # Inner Numeric Constraints -- used to floating point under or overflow
      :floorToPreventOverflow => 1e-30
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

