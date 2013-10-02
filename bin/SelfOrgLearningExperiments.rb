### VERSION "nCore"
## ../nCore/bin/SelfOrgLearningExperiments.rb
# Purpose:  To quantitatively explore the simplest clustering w/o supervision.
# This is a simplified and significantly reorganized version of 'Phase1Phase2MultiCycle.rb'

require_relative 'BaseLearningExperiment'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

###################################### START of Main Learning  ##########################################

class Experiment

  def setParameters

    numberOfExamples = 4
    randomNumberSeed = 0

    @args = {
        :experimentNumber => ExperimentLogger.number,
        :descriptionOfExperiment => descriptionOfExperiment,
        :rng => Random.new(randomNumberSeed),

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.02,
        :minMSE => 0.0001,
        :maxNumEpochs => 50,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 0,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => numberOfExamples,

        # Recording and database parameters
        :intervalForSavingNeuronData => 500,
        :intervalForSavingDetailedNeuronData => 1,
        :intervalForSavingTrainingData => 500,

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
        ##  :symmetricalCenters => true, # if true, speed is negatively affected
        :alwaysUseFuzzyClusters => true,
        :epochsBeforeFlockingAllowed => 200,

        # Inner Numeric Constraints -- used to floating point under or overflow
        :floorToPreventOverflow => 1e-30
    }
  end

  def createTrainingSet(args)
    examples = []
    examples << {:inputs => [1.0, 1.0], :targets => [0.0], :exampleNumber => 0, :class => 0}
    examples << {:inputs => [1.0, -1.0], :targets => [1.0], :exampleNumber => 1, :class => 1}
    examples << {:inputs => [-1.0, -1.0], :targets => [0.0], :exampleNumber => 2, :class => 0}
    examples << {:inputs => [-1.0, 1.0], :targets => [1.0], :exampleNumber => 3, :class => 1}
    if (args[:numberOfExamples] != examples.length)
      STDERR.puts "****************Incorrect Number of Examples Specified!! ************************"
    end
    examples
  end

  def createNetworkAndTrainer(examples)
    network = SimpleFlockingNetwork.new(args)
    theTrainer = TrainingSupervisorOutputNeuronLocalFlocking.new(examples, network, args)
    return network, theTrainer
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
    keysToRecords.reject! { |recordKey| recordKey.empty? }
    neuronDataRecords = keysToRecords.collect { |recordKey| NeuronData.values(recordKey) } unless (keysToRecords.empty?)
    puts neuronDataRecords


    puts "\n\n############ DetailedNeuronData #############\n"
    puts "Epoch\t\t\t\tEx0\t\t\t\tEx1\t\t\t\tEx2\t\t\t\tEx3"
    keysToRecords = []
    DetailedNeuronData.lookup_values(:epochs).each do |epochNumber|
      netInputs = []
      DetailedNeuronData.lookup_values(:exampleNumber).each do |anExampleNumber|
        aRecordKey = DetailedNeuronData.lookup { |q| q[:experimentNumber_epochs_neuron_exampleNumber].eq({experimentNumber: ExperimentLogger.number, epochs: epochNumber, neuron: neuronToDisplay, exampleNumber: anExampleNumber}) }
        unless (aRecordKey.empty?)
          aHash = DetailedNeuronData.values(aRecordKey)
          netInputs << aHash[:netInput]
        end
      end

      aStringToPrint = netInputs.join("\t\t")
      puts "#{epochNumber}\t\t#{aStringToPrint}"
    end

    #unless (keysToRecords.empty?)
    #  keysToRecords.reject! { |recordKey| recordKey.empty? }
    #  # keysToRecords.each { |recordKey| std("recordKey", recordKey) } unless (keysToRecords.empty?) # TODO DEBUG CODE
    #  detailedNeuronDataRecords = keysToRecords.collect { |recordKey| DetailedNeuronData.values(recordKey) }
    #
    #
    #
    #end
    #
    #puts detailedNeuronDataRecords
    #

    puts "\n\n############ TrainingData #############"
    keysToRecords = TrainingData.lookup { |q| q[:experimentNumber].eq({experimentNumber: ExperimentLogger.number}) }
    keysToRecords.reject! { |recordKey| recordKey.empty? }
    trainingDataRecords = keysToRecords.collect { |recordKey| TrainingData.values(recordKey) } unless (keysToRecords.empty?)
    puts trainingDataRecords


    puts "\n\n############ SnapShotData #############"
    dataToStoreLongTerm = {:experimentNumber => ExperimentLogger.number, :descriptionOfExperiment => descriptionOfExperiment,
                           :network => network, :time => Time.now, :elapsedTime => (Time.now - startingTime),
                           :epochs => lastEpoch, :trainMSE => lastTrainingMSE, :testMSE => lastTestingMSE,
                           :accumulatedAbsoluteFlockingErrors => accumulatedAbsoluteFlockingErrors
    }
    SnapShotData.new(dataToStoreLongTerm)

    keysToRecords = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(5) }
    keysToRecords.reject! { |recordKey| recordKey.empty? }
    unless (keysToRecords.empty?)
      puts
      puts "Number\tLastEpoch\tTrainMSE\t\tTestMSE\t\tTime\t\t\t\tDescription"
      keysToRecords.each do |keyToOneRecord|
        recordHash = SnapShotData.values(keyToOneRecord)
        puts "#{recordHash[:experimentNumber]}\t\t#{recordHash[:epochs]}\t#{recordHash[:trainMSE]}\t#{recordHash[:testMSE]}\t\t#{recordHash[:time]}\t\t\t\t#{recordHash[:descriptionOfExperiment]}"
      end

      # recordHash = SnapShotData.values(keysToRecords.last)
    end

    plotMSEvsEpochNumber(trainingDataRecords)
  end

end

experiment = Experiment.new("T2LearningRateExperiments using correctionFactorForRateAtWhichNeuronsGainChanges")

experiment.performSimulation()
