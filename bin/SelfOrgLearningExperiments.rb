### VERSION "nCore"
## ../nCore/bin/SelfOrgLearningExperiments.rb
# Purpose:  To quantitatively explore the simplest clustering w/o supervision.

require_relative 'BaseLearningExperiment'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

###################################### START of Main Learning  ##########################################

class Experiment

  def setParameters
    @args = {
        :experimentNumber => ExperimentLogger.number,
        :descriptionOfExperiment => descriptionOfExperiment,
        :rng => Random.new(randomNumberSeed),

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.02,
        :minMSE => 0.0001,
        :maxNumEpochs => 10,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 0,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => (numberOfExamples = 4),

        # Recording and database parameters
        :intervalForSavingNeuronData => 50,
        :intervalForSavingDetailedNeuronData => 1,
        :intervalForSavingTrainingData => 50,

        # Flocking Parameters...
        :flockingLearningRate => -0.1, # -0.002
        :maxFlockingIterationsCount => 2000, # 2000,
        :maxAbsFlockingErrorsPerExample => 0.002, #0.002, # 0.005,   # 0.04 / numberOfExamples = 0.005

        :typeOfClusterer => DynamicClusterer,
        :numberOfClusters => 2,
        :m => 2.0,
        :numExamples => numberOfExamples,
        :exampleVectorLength => 1,
        :delta => 1e-2,
        :maxNumberOfClusteringIterations => 10,
        :keepTargetsSymmetrical => true,
        :targetDivergenceFactor => 1.0,
        :alwaysUseFuzzyClusters => true,
        #  :epochsBeforeFlockingAllowed => 200,  DNA

        # Inner Numeric Constraints -- used to floating point under or overflow
        :floorToPreventOverflow => 1e-60 # 1e-30
    }
  end

  def createTrainingSet
    examples = []
    examples << {:inputs => [1.0, 1.0], :targets => [0.0], :exampleNumber => 0, :class => 0}
    examples << {:inputs => [1.0, -1.0], :targets => [1.0], :exampleNumber => 1, :class => 1}
    examples << {:inputs => [-1.0, -1.0], :targets => [0.0], :exampleNumber => 2, :class => 0}
    examples << {:inputs => [-1.0, 1.0], :targets => [1.0], :exampleNumber => 3, :class => 1}
    self.numberOfExamples = examples.length
    return examples
  end

  def createNetworkAndTrainer
    network = Flocking1LayerNetwork.new(args)
    theTrainer = TrainingSuperONLYLocalFlocking.new(examples, network, args)
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
    epochsArray = []
    examplesNetInputs = []
    DetailedNeuronData.lookup_values(:epochs).each do |epochNumber|
      netInputs = []
      epochsArray << epochNumber.to_i
      DetailedNeuronData.lookup_values(:exampleNumber).each do |anExampleNumber|
        aRecordKey = DetailedNeuronData.lookup { |q| q[:experimentNumber_epochs_neuron_exampleNumber].eq({experimentNumber: ExperimentLogger.number, epochs: epochNumber, neuron: neuronToDisplay, exampleNumber: anExampleNumber}) }
        unless (aRecordKey.empty?)

          aHash = DetailedNeuronData.values(aRecordKey)
          aNetInput = aHash[:netInput]
          exampleNumber = anExampleNumber.to_i
          examplesNetInputs[exampleNumber] = [] if (examplesNetInputs[exampleNumber].nil?)
          examplesNetInputs[exampleNumber] << aNetInput
          netInputs << aNetInput
        end
      end

      aStringToPrint = netInputs.join("\t\t")
      puts "#{epochNumber}\t\t#{aStringToPrint}"
    end

    aPlotter = Plotter.new(title="Inputs to Neuron vs. Time", "Number of Epochs", "Net Input to Neuron", plotOutputFilenameBase = "#{Dir.home}/Code/Ruby/NN2012/plots/netInputVsEpochs")
    aPlotter.plotNetInputs(epochsArray, examplesNetInputs)


    puts "\n\n############ TrainingData #############"
    keysToRecords = TrainingData.lookup { |q| q[:experimentNumber].eq({experimentNumber: ExperimentLogger.number}) }
    keysToRecords.reject! { |recordKey| recordKey.empty? }
    trainingDataRecords = keysToRecords.collect { |recordKey| TrainingData.values(recordKey) } unless (keysToRecords.empty?)
    puts trainingDataRecords

    theOutputNeuron = network.allNeuronLayers[1][0]

    puts "\n\n############ SnapShotData #############"
    dataToStoreLongTerm = {:experimentNumber => ExperimentLogger.number, :descriptionOfExperiment => descriptionOfExperiment,
                           :network => network, :time => Time.now, :elapsedTime => (Time.now - startingTime),
                           :epochs => lastEpoch, :trainMSE => lastTrainingMSE, :testMSE => lastTestingMSE,
                           :accumulatedAbsoluteFlockingErrors => accumulatedAbsoluteFlockingErrors,
                           :examplesInClusters => theOutputNeuron.examplesContainedInEachCluster
    }
    SnapShotData.new(dataToStoreLongTerm)

    keysToRecords = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(10) }
    keysToRecords.reject! { |recordKey| recordKey.empty? }
    unless (keysToRecords.empty?)
      puts
      puts "Number\tLastEpoch\tTrainMSE\t\tTestMSE\t\t\tAccumulatedAbsoluteFlockingErrors\tTime\t\t\t\t\t\t\tDescription"
      keysToRecords.each do |keyToOneRecord|
        recordHash = SnapShotData.values(keyToOneRecord)
        puts "#{recordHash[:experimentNumber]}\t\t#{recordHash[:epochs]}\t\t\t#{recordHash[:trainMSE]}\t#{recordHash[:testMSE]}\t\t\t#{recordHash[:accumulatedAbsoluteFlockingErrors][0]}\t\t\t\t#{recordHash[:time]}\t\t#{recordHash[:descriptionOfExperiment]}\t\t#{recordHash[:examplesInClusters]}"
      end

      # recordHash = SnapShotData.values(keysToRecords.last)
    end

    plotMSEvsEpochNumber(trainingDataRecords)
  end

end

seed = 0
#20.times do |seed|
  rSeed = seed + 100
  experiment = Experiment.new("SelfOrgLearningRateExperiments using correctionFactorForRateAtWhichNeuronsGainChanges", randomNumberSeed = rSeed)
  experiment.performSimulation()
#end
