### VERSION "nCore"
## ../nCore/bin/DeepRecurrent.rb

require_relative 'BaseLearningExperiment'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

class Experiment

  def setParameters

    args = {
        :experimentNumber => experimentLogger.experimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.3, # 0.03,
        :minMSE => 0.001,
        :maxNumEpochs => 2e3, # 120e3,

        # Network Architecture and initial weights
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 1,
        :numberOfHiddenLayers => 1,
        :numberOfOutputNeurons => 2,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => (numberOfExamples = 8),

        # Recording and database parameters
        :intervalForSavingNeuronData => 10000,
        :intervalForSavingDetailedNeuronData => 10000,
        :intervalForSavingTrainingData => 1000,

        # Flocking Parameters...
        :flockingLearningRate => -0.009, # -0.0002,
        :bpFlockingLearningRate => -0.063,
        :maxFlockingIterationsCount => 60, # 2000,
        :targetFlockIterationsCount => 20,
        :ratioDropInMSE => 0.95,
        :ratioDropInMSEForFlocking => 0.96,
        # :maxAbsFlockingErrorsPerExample => 0.2, #  0.04 / numberOfExamples = 0.005
        :ratioDropInMSEForBPFlocking => 0.98,
        :maxBPFlockingIterationsCount => 60,
        :targetBPFlockIterationsCount => 20,


        # Flocker Specs...
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
        :epochsBeforeFlockingAllowed => 0, #  10e1,

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


  def createTestingSet
    createTestingSet
  end


  def createNetworkAndTrainer
    network = DeepRecurrentNetwork.new(args)
    theTrainer = TrainSuperCircleProblemBPFlockAndLocFlockAtOutputNeuron.new(examples, network, args)
    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

randomNumberSeed = 2

experiment = Experiment.new("J1DeepRecurrent", randomNumberSeed)

experiment.performSimulation()
