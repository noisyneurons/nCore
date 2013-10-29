### VERSION "nCore"
## ../nCore/bin/T4LearningRateExperiments.rb

require_relative 'BaseLearningExperiment'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

class Experiment
  def setParameters

    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :rng => Random.new(randomNumberSeed),

        # training parameters re. Output Error
        :outputErrorLearningRate => 1.0, # 0.4, #0.02,
        :minMSE => 0.0001,
        :maxNumEpochs => 4e3,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 1,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => (numberOfExamples = 8),

        # Recording and database parameters
        :intervalForSavingNeuronData => 100,
        :intervalForSavingDetailedNeuronData => 1000,
        :intervalForSavingTrainingData => 100,

        # Flocking Parameters...
        :flockingLearningRate => -0.002, # -0.002,
        :maxFlockingIterationsCount => 10, # 2000,
        :maxAbsFlockingErrorsPerExample => 0.002, #  0.04 / numberOfExamples = 0.005

        # Flocker Specs
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
        :epochsBeforeFlockingAllowed => 200,

        # Inner Numeric Constraints -- used to floating point under or overflow
        :floorToPreventOverflow => 1e-60 # 1e-30
    }
  end

  def createTrainingSet
    examples = []
    examples << {:inputs => [1.0, 1.0], :targets => [1.0], :exampleNumber => 0, :class => 1}
    examples << {:inputs => [1.0, 2.0], :targets => [1.0], :exampleNumber => 1, :class => 1}
    examples << {:inputs => [1.0, 3.0], :targets => [1.0], :exampleNumber => 2, :class => 1}
    examples << {:inputs => [1.0, 4.0], :targets => [1.0], :exampleNumber => 3, :class => 1}
    examples << {:inputs => [-1.0, -1.0], :targets => [0.0], :exampleNumber => 4, :class => 0}
    examples << {:inputs => [-1.0, -2.0], :targets => [0.0], :exampleNumber => 5, :class => 0}
    examples << {:inputs => [-1.0, -3.0], :targets => [0.0], :exampleNumber => 6, :class => 0}
    examples << {:inputs => [-1.0, -4.0], :targets => [0.0], :exampleNumber => 7, :class => 0}
    STDERR.puts "Error: Incorrect Number of Examples Generated and/or Specified" unless (examples.length == args[:numberOfExamples])
    return examples
  end

  def createNetworkAndTrainer
    network = Flocking3LayerNetwork.new(args)
    theTrainer = TrainingSupervisorAllLocalFlockingLayers.new(examples, network, args)
    return network, theTrainer
  end
end

###################################### START of Main Learning  ##########################################

experiment = Experiment.new("T4LearningRateExperiments using correctionFactorForRateAtWhichNeuronsGainChanges", randomNumberSeed=0)

experiment.performSimulation()
