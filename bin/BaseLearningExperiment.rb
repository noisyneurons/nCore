### VERSION "nCore"
## ../nCore/bin/BaseLearningExperiments.rb

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralPartsExtended'
# require_relative '../lib/core/ExampleImportanceMods'    # TODO Is this useful???  So far NOT!
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/TrainingSequencingAndGrouping'
require_relative '../lib/core/Trainers.rb'

class Experiment
  attr_accessor :descriptionOfExperiment, :randomNumberSeed, :experimentLogger, :examples, :numberOfExamples, :args, :trainingSequence
  include ExampleDistribution

  def initialize(descriptionOfExperiment, randomNumberSeed)
    @descriptionOfExperiment = descriptionOfExperiment
    @randomNumberSeed = randomNumberSeed
    srand(randomNumberSeed)
    @experimentLogger = ExperimentLogger.new(descriptionOfExperiment)
    @args = self.setParameters
    @examples = createTrainingSet
    @trainingSequence = TrainingSequence.new(args)
    @args[:trainingSequence] = trainingSequence
  end

  def setParameters
    @args = {
        :experimentNumber => ExperimentLogger.number,
        :descriptionOfExperiment => descriptionOfExperiment,
        :rng => Random.new(randomNumberSeed),

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
        :intervalForSavingNeuronData => 100,
        :intervalForSavingDetailedNeuronData => 1000,
        :intervalForSavingTrainingData => 100,

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
        :keepTargetsSymmetrical => true,
        :alwaysUseFuzzyClusters => true,
        :epochsBeforeFlockingAllowed => 200,

        # Inner Numeric Constraints -- used to floating point under or overflow
        :floorToPreventOverflow => 1e-60
    }
  end

  def createTrainingSet
    STDERR.puts "Error: base class method called!!"
    STDERR.puts "Error: Incorrect Number of Examples Generated and/or Specified" unless(examples.length == args[:numberOfExamples])
    return examples
  end

  def reportTrainingResults(neuronToDisplay, accumulatedAbsoluteFlockingErrors, descriptionOfExperiment, lastEpoch, lastTrainingMSE, network, startingTime)
    puts network

    lastTestingMSE = nil
    puts "lastEpoch, lastTrainingMSE, accumulatedAbsoluteFlockingErrors, lastTestingMSE"
    puts lastEpoch, lastTrainingMSE, accumulatedAbsoluteFlockingErrors, lastTestingMSE

    puts "\n\n############ NeuronData #############"
    keysToRecords = []
    NeuronData.lookup_values(:epochs).each do |epochNumber|
      keysToRecords << NeuronData.lookup { |q| q[:experimentNumber_epochs_neuron].eq({experimentNumber: ExperimentLogger.number, epochs: epochNumber, neuron: neuronToDisplay}) }
    end
    neuronDataRecords = nil
    unless (keysToRecords.empty?)
      keysToRecords.reject! { |recordKey| recordKey.empty? }
      neuronDataRecords = keysToRecords.collect { |recordKey| NeuronData.values(recordKey) }
    end
    puts neuronDataRecords


    puts "\n\n############ DetailedNeuronData #############"
    keysToRecords = []
    DetailedNeuronData.lookup_values(:epochs).each do |epochNumber|
      DetailedNeuronData.lookup_values(:exampleNumber).each do |anExampleNumber|
        keysToRecords << DetailedNeuronData.lookup { |q| q[:experimentNumber_epochs_neuron_exampleNumber].eq({experimentNumber: ExperimentLogger.number, epochs: epochNumber, neuron: neuronToDisplay, exampleNumber: anExampleNumber}) }
      end
    end
    unless (keysToRecords.empty?)
      keysToRecords.reject! { |recordKey| recordKey.empty? }
      detailedNeuronDataRecords = keysToRecords.collect { |recordKey| DetailedNeuronData.values(recordKey) }
    end
    puts detailedNeuronDataRecords


    puts "\n\n############ TrainingData #############"
    keysToRecords = TrainingData.lookup { |q| q[:experimentNumber].eq({experimentNumber: ExperimentLogger.number}) }
    trainingDataRecords = nil
    unless (keysToRecords.empty?)
      keysToRecords.reject! { |recordKey| recordKey.empty? }
      trainingDataRecords = keysToRecords.collect { |recordKey| TrainingData.values(recordKey) }
    end
    puts trainingDataRecords


    puts "\n\n############ SnapShotData #############"
    dataToStoreLongTerm = {:experimentNumber => ExperimentLogger.number, :descriptionOfExperiment => descriptionOfExperiment,
                           :network => network, :time => Time.now, :elapsedTime => (Time.now - startingTime),
                           :epochs => lastEpoch, :trainMSE => lastTrainingMSE, :testMSE => lastTestingMSE,
                           :accumulatedAbsoluteFlockingErrors => accumulatedAbsoluteFlockingErrors
    }
    SnapShotData.new(dataToStoreLongTerm)

    keysToRecords = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(10) }
    unless (keysToRecords.empty?)
      puts
      puts "Number\tLastEpoch\tTrainMSE\t\tTestMSE\t\t\tAccumulatedAbsoluteFlockingErrors\t\t\tTime\t\t\t\tDescription"
      keysToRecords.each do |keyToOneRecord|
        recordHash = SnapShotData.values(keyToOneRecord)
        puts "#{recordHash[:experimentNumber]}\t\t#{recordHash[:epochs]}\t\t#{recordHash[:trainMSE]}\t#{recordHash[:testMSE]}\t\t\t#{recordHash[:accumulatedAbsoluteFlockingErrors]}\t\t\t#{recordHash[:time]}\t\t\t\t#{recordHash[:descriptionOfExperiment]}"
      end

      # recordHash = SnapShotData.values(keysToRecords.last)
    end

    plotMSEvsEpochNumber(trainingDataRecords)
  end

  def performSimulation

######################## Create Network and Trainer ....
    network, theTrainer = createNetworkAndTrainer

###################################### perform Learning/Training  ##########################################

    startingTime = Time.now
    lastEpoch, lastTrainingMSE, accumulatedAbsoluteFlockingErrors = theTrainer.train

############################## reporting results....

    neuronToDisplay = 2
    reportTrainingResults(neuronToDisplay, accumulatedAbsoluteFlockingErrors, descriptionOfExperiment, lastEpoch, lastTrainingMSE, network, startingTime)

############################## clean-up....
    experimentLogger.deleteTemporaryDataRecordsInDB()
    experimentLogger.save
  end
end

