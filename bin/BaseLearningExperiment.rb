### VERSION "nCore"
## ../nCore/bin/BaseLearningExperiments.rb

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers.rb'

class Experiment
  attr_accessor :network, :theTrainer, :descriptionOfExperiment, :taskID, :jobID, :jobName, :randomNumberSeed,
                :experimentLogger, :simulationDataStoreManager, :examples, :numberOfExamples, :args, :trainingSequence
  include ExampleDistribution

  def initialize(descriptionOfExperiment, baseRandomNumberSeed)
    @descriptionOfExperiment = descriptionOfExperiment
    @taskID = ((ENV['SGE_TASK_ID']).to_i) || 0
    @randomNumberSeed = baseRandomNumberSeed + (taskID * 10000)
    srand(randomNumberSeed)

    puts "sleeping" unless ($currentHost == "localhost")
    sleep(rand * 30) unless ($currentHost == "localhost")

    @jobID = ((ENV['JOB_ID']).to_i) || 0
    @jobName = descriptionOfExperiment[0...10]

    @experimentLogger = ExperimentLogger.new(descriptionOfExperiment, jobName)
    $globalExperimentNumber = experimentLogger.experimentNumber
    @args = self.setParameters
    @examples = createTrainingSet
    args[:testingExamples] = createTestingSet

    @trainingSequence = args[:trainingSequence].new(args)

    @simulationDataStoreManager = SimulationDataStoreManager.new(args)
    @args[:trainingSequence] = trainingSequence
  end

  def setParameters

    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.02,
        :minMSE => 0.0001,
        :maxNumEpochs => 4e3,
        :numLoops => 10,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 0,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = nil),

        # Recording, Display, and Database parameters
        :neuronsToDisplay => [5],
        :intervalForSavingNeuronData => 100,
        :intervalForSavingDetailedNeuronData => 1000,
        :intervalForSavingTrainingData => 100,

        # Inner Numeric Constraints -- used to floating point under or overflow
        :floorToPreventOverflow => 1e-60
    }
  end

  def createTrainingSet
    STDERR.puts "Error: base class method called!!"
    STDERR.puts "Error: Incorrect Number of Examples Generated and/or Specified" unless (examples.length == args[:numberOfExamples])
    return examples
  end

  def createTestingSet
    return nil
  end

  def reportTrainingResults(neuronToDisplay, descriptionOfExperiment, lastEpoch, lastTrainingMSE, lastTestingMSE, network, startingTime)

    endOfTrainingReport(lastEpoch, lastTestingMSE, lastTrainingMSE, network)

    #neuronDataSummary(neuronToDisplay)

    #detailedNeuronDataSummary(neuronToDisplay)

    trainingDataRecords = trainingDataSummary

    storeSnapShotData(descriptionOfExperiment, lastEpoch, lastTestingMSE, lastTrainingMSE, network, startingTime)

    snapShotDataSummary

    plotMSEvsEpochNumber(trainingDataRecords)

    # plotTrainingResults(neuronToDisplay)
  end

  def performSimulation

######################## Create Network and Trainer ....
    self.network, self.theTrainer = createNetworkAndTrainer

###################################### perform Learning/Training  ##########################################

    startingTime = Time.now
    lastEpoch, lastTrainingMSE, lastTestingMSE = theTrainer.train

############################## reporting results....

    reportTrainingResults(args[:neuronsToDisplay], descriptionOfExperiment,
                          lastEpoch, lastTrainingMSE, lastTestingMSE, network, startingTime)

############################## clean-up....
    simulationDataStoreManager.deleteTemporaryDataRecordsInDB(experimentLogger.experimentNumber)
    simulationDataStoreManager.save
  end

  # routines supporting 'reportTrainingResults':

  def endOfTrainingReport(lastEpoch, lastTestingMSE, lastTrainingMSE, network)
    puts "\n\n_________________________________________________________________________________________________________"
    STDERR.puts "ExperimentNumber=\t#{$globalExperimentNumber}"
    puts "ExperimentNumber=\t#{$globalExperimentNumber}"
    puts " GridEngineJobID= \t#{jobID}\n\n"

    puts network

    puts "lastEpoch, lastTrainingMSE, lastTestingMSE"
    puts lastEpoch, lastTrainingMSE, lastTestingMSE
  end

  def neuronDataSummary(neuronsToDisplay)
    puts "\n\n############ NeuronData #############"
    neuronsToDisplay.each do |neuronToDisplay|
      keysToRecords = []
      NeuronData.lookup_values(:epochs).each do |epochNumber|
        keysToRecords << NeuronData.lookup { |q| q[:experimentNumber_epochs_neuron].eq({experimentNumber: $globalExperimentNumber, epochs: epochNumber, neuron: neuronToDisplay}) }
      end
      # keysToRecords = NeuronData.lookup { |q| q[:experimentNumber_epochs_neuron].eq({experimentNumber: $globalExperimentNumber, epochs: 899, neuron: neuronToDisplay}) }

      puts "NeuronData number of Records Retrieved= #{keysToRecords.length}"
      neuronDataRecords = nil
      unless (keysToRecords.empty?)
        # keysToRecords.each { |recordKey| puts "empty" if(recordKey.empty?) }
        keysToRecords.reject! { |recordKey| recordKey.empty? }
        neuronDataRecords = keysToRecords.collect { |recordKey| NeuronData.values(recordKey) }
      end
      puts neuronDataRecords
      puts "\n"
    end
  end

  def detailedNeuronDataSummary(neuronsToDisplay)
    puts "\n\n############ DetailedNeuronData #############"
    neuronsToDisplay.each do |neuronToDisplay|
      keysToRecords = []
      DetailedNeuronData.lookup_values(:epochs).each do |epochNumber|
        DetailedNeuronData.lookup_values(:exampleNumber).each do |anExampleNumber|
          keysToRecords << DetailedNeuronData.lookup { |q| q[:experimentNumber_epochs_neuron_exampleNumber].eq({experimentNumber: $globalExperimentNumber, epochs: epochNumber, neuron: neuronToDisplay, exampleNumber: anExampleNumber}) }
        end
      end
      puts "DetailedNeuronData number of Records Retrieved= #{keysToRecords.length}"
      unless (keysToRecords.empty?)
        keysToRecords.reject! { |recordKey| recordKey.empty? }
        # puts "DetailedNeuronData keysToRecords=\t#{keysToRecords}"
        detailedNeuronDataRecords = keysToRecords.flatten.collect { |recordKey| DetailedNeuronData.values(recordKey) }
      end
      puts detailedNeuronDataRecords
      puts "\n"
    end
  end

  def trainingDataSummary
    puts "\n\n############ TrainingData #############"
    keysToRecords = TrainingData.lookup { |q| q[:experimentNumber].eq({experimentNumber: $globalExperimentNumber}) }
    puts "TrainingData number of Records Retrieved= #{keysToRecords.length}"

    trainingDataRecords = nil
    unless (keysToRecords.empty?)
      keysToRecords.reject! { |recordKey| recordKey.empty? }
      trainingDataRecords = keysToRecords.collect { |recordKey| TrainingData.values(recordKey) }
    end
    puts trainingDataRecords
    trainingDataRecords
  end

  def storeSnapShotData(descriptionOfExperiment, lastEpoch, lastTestingMSE, lastTrainingMSE, network, startingTime)
    puts "\n\n############ SnapShotData #############"
    dataToStoreLongTerm = {:experimentNumber => $globalExperimentNumber, :descriptionOfExperiment => descriptionOfExperiment,
                           :gridTaskID => self.taskID, :gridJobID => self.jobID, :network => network, :args => args,
                           :time => Time.now, :elapsedTime => (Time.now - startingTime),
                           :epochs => lastEpoch, :trainMSE => lastTrainingMSE, :testMSE => lastTestingMSE
    }
    SnapShotData.new(dataToStoreLongTerm)
  end

  def snapShotDataSummary
    keysToRecords = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(2) }
    unless (keysToRecords.empty?)
      puts
      puts "Number\tLastEpoch\t\tTrainMSE\t\t\tTestMSE\t\t\tAccumulatedAbsoluteFlockingErrors\t\t\t\tTime\t\tTaskID\t\t\t\t\tDescription"
      keysToRecords.each do |keyToOneRecord|
        begin
          recordHash = SnapShotData.values(keyToOneRecord)
          puts "#{recordHash[:experimentNumber]}\t\t#{recordHash[:epochs]}\t\t#{recordHash[:trainMSE]}\t\t#{recordHash[:testMSE]}\t\t\t#{recordHash[:accumulatedAbsoluteFlockingErrors]}\t\t\t#{recordHash[:time]}\t\t\t#{recordHash[:gridTaskID]}\t\t\t\t#{recordHash[:descriptionOfExperiment]}"
        rescue
          puts "problem in yaml conversion"
        end
      end
      # recordHash = SnapShotData.values(keysToRecords.last)
    end
  end

  def plotTrainingResults(arrayOfNeuronsToPlot)
    generatePlotForEachNeuron(arrayOfNeuronsToPlot) if arrayOfNeuronsToPlot.present?
  end
end

