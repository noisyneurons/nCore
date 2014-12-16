### VERSION "nCore"
## ../nCore/bin/LearningRateExperiments.rb
# Purpose:  To quantitatively explore the simplest clustering w/o supervision.
# This is a simplified and significantly reorganized version of 'Phase1Phase2MultiCycle.rb

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralPartsExtended'
# require_relative '../lib/core/ExampleImportanceMods'    # TODO where should this go?
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers.rb'
# require_relative '../lib/core/ExampleImportanceMods'    # TODO where should this go?
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'


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
  logger.puts "****************Incorrect Number of Examples Specified!! ************************" if (args[:numberOfExamples] != examples.length)
  return examples
end

def displayAndPlotResults(args, dPrimes, dataStoreManager, lastEpoch,
    lastTestingMSE, lastTrainingMSE, network, theTrainer, trainingSequence)
  logger.puts network
  logger.puts "Elapsed Time=\t#{theTrainer.elapsedTime}"
  logger.puts "\tAt Epoch #{trainingSequence.epochs}"
  logger.puts "\tAt Epoch #{lastEpoch}"
  logger.puts "\t\tThe Network's Training MSE=\t#{lastTrainingMSE}\t and TEST MSE=\t#{lastTestingMSE}\n"
  logger.puts "\t\t\tThe dPrime(s) at the end of training are: #{dPrimes}"

#############################  plotting and visualization....
  plotMSEvsEpochNumber(network)
end

class Experiment
  def setParameters

    numberOfExamples = 8
    randomNumberSeed = 0

    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :rng => Random.new(randomNumberSeed),

        :phase1Epochs => 10000,
        :phase2Epochs => 0,

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.02,
        :minMSE => 0.0001,
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
        :flockingLearningRate => -0.002,
        :maxFlockingIterationsCount => 2000, # 3800, # 2000,
        :maxAbsFlockingErrorsPerExample => 0.002, # 0.00000000000001, #0.002, # 0.005,   # 0.04 / numberOfExamples = 0.005

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
end

###################################### START of Main Learning  ##########################################
######################## Specify data store and experiment description....

srand(0)
descriptionOfExperiment = "SimpleAdjustableLearningRateTrainerMultiFlockIterations Reference Run NUMBER 2"
experiment = ExperimentLogger.new(descriptionOfExperiment)
args = experiment.setParameters
# dataStoreManager = SimulationDataStoreManager.create
args[:dataStoreManager] = dataStoreManager = SimulationDataStoreManager.new(args)
args[:trainingSequence] = trainingSequence = TrainingSequence.new(args)

############################### create training set...
examples = createTrainingSet(args)

######################## Create Network....
network = Flocking1LayerNetwork.new(dataStoreManager, args) # TODO Currently need to insure that TrainingSequence.create has been called before network creation!!!
logger.puts network

############################### Create Trainer ...
theTrainer = SimpleAdjustableLearningRateTrainer.new(trainingSequence, network, args)

arrayOfNeuronsToPlot = nil
lastEpoch, lastTrainingMSE, accumulatedAbsoluteFlockingErrors = theTrainer.simpleLearningWithFlocking(examples)
theTrainer.displayTrainingResults(arrayOfNeuronsToPlot)

lastTestingMSE = nil
# theTrainer.storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, accumulatedAbsoluteFlockingErrors)

###################################### END of Main Learning ##########################################


logger.puts "############ Include Example Numbers #############"
4000.times do |epochNumber|
  selectedData = DetailedNeuronData.lookup { |q| q[:experimentNumber_epochs_neuron].eq({experimentNumber: $globalExperimentNumber, epochs: epochNumber,
                                                                                        neuron: 2}) }
  logger.puts "For epoch number=\t#{epochNumber}" unless (selectedData.empty?)

  selectedData.each { |itemKey| logger.puts DetailedNeuronData.values(itemKey) } unless (selectedData.empty?)

end
logger.puts "####################################"

displayAndPlotResults(args, accumulatedAbsoluteFlockingErrors, dataStoreManager, lastEpoch, lastTestingMSE,
                      lastTrainingMSE, network, theTrainer, trainingSequence)

SnapShotData.new(descriptionOfExperiment, network, Time.now, lastEpoch, lastTrainingMSE, lastTestingMSE)

selectedData = SnapShotData.lookup { |q| q[:experimentNumber_epochs].eq({experimentNumber: $globalExperimentNumber, epochs: lastEpoch}) }

selectedData = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(5) }
unless (selectedData.empty?)
  logger.puts
  logger.puts "Number\tDescription\tLastEpoch\tTrainMSE\tTestMSE\tTime"
  selectedData.each do |aSelectedExperiment|
    aHash = SnapShotData.values(aSelectedExperiment)
    logger.puts "#{aHash[:experimentNumber]}\t#{aHash[:descriptionOfExperiment]}\t#{aHash[:epochs]}\t#{aHash[:trainMSE]}\t#{aHash[:testMSE]}\t#{aHash[:time]}"
  end
end

DetailedNeuronData.deleteData($globalExperimentNumber)
NeuronData.deleteData($globalExperimentNumber)

experiment.save

