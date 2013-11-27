### VERSION "nCore"
## ../nCore/bin/ThreeClass2HiddenLayersLocalFlock.rb
## Simple backprop demo. For XOR, and given parameters, requires 2080 epochs to converge.
##                       For OR, and given parameters, requires 166 epochs to converge.

require_relative 'BaseLearningExperiment'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

class Experiment
  include NeuronToNeuronConnection

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters re. Output Error
        :outputErrorLearningRate => 1.0,
        :minMSE => 0.00001,
        :maxNumEpochs => 1e3,
        :numLoops => 500,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenLayer1Neurons => 2,
        :numberOfHiddenLayer2Neurons => 2,
        :numberOfOutputNeurons => 3,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => (numExamples = 6),
        :numExamples => numExamples,

        # Recording and database parameters
        :neuronToDisplay => 3,
        :intervalForSavingNeuronData => 10000,
        :intervalForSavingDetailedNeuronData => 700,
        :intervalForSavingTrainingData => 100,

        # Flocking Parameters...
        :flockingLearningRate => -0.01, # -0.01, # -0.0002,
        :maxFlockingIterationsCount => 300, # 2000,
        :targetFlockIterationsCount => 20,
        :ratioDropInMSE => 0.95, # 0.01, # 0.95,
        :ratioDropInMSEForFlocking => 0.97, # 0.015, # 0.97,

        # Flocker Specs...
        :typeOfClusterer => DynamicClusterer,
        :numberOfClusters => 2,
        :m => 2.0,
        :numExamples => numExamples,
        :exampleVectorLength => 1,
        :delta => 1e-2,
        :maxNumberOfClusteringIterations => 30,
        :keepTargetsSymmetrical => true,
        :targetDivergenceFactor => 1.0,
        :alwaysUseFuzzyClusters => true,
        # :epochsBeforeFlockingAllowed => 300, #  10e1, no longer used??
        :maxLargestEuclidianDistanceMovedThatIsWOErrorMsg => 0.01,

        # Inner Numeric Constraints -- used to floating point under or overflow
        :floorToPreventOverflow => 1e-60 # 1e-30
    }
  end

  def createTrainingSet
    examples = []
    examples << {:inputs => [0.0, 0.0], :targets => [1.0, 0.0, 0.0], :exampleNumber => 0, :class => 0}
    examples << {:inputs => [1.0, 0.0], :targets => [1.0, 0.0, 0.0], :exampleNumber => 1, :class => 0}
    examples << {:inputs => [0.0, 1.0], :targets => [0.0, 1.0, 0.0], :exampleNumber => 2, :class => 1}
    examples << {:inputs => [1.0, 1.0], :targets => [0.0, 1.0, 0.0], :exampleNumber => 3, :class => 1}
    examples << {:inputs => [0.0, 2.0], :targets => [0.0, 0.0, 1.0], :exampleNumber => 4, :class => 2}
    examples << {:inputs => [1.0, 2.0], :targets => [0.0, 0.0, 1.0], :exampleNumber => 5, :class => 2}
    return examples
  end

  def createNetworkAndTrainer
    network = Recurrent2HiddenLayerNetworkSpecial.new(args)
    # puts network.to_s
    theTrainer = ThreeClass2HiddenSupervisorLocalFlock.new(examples, network, args)
    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("3Cls2HidLocal ThreeClass2HiddenLayersLocalFlock", baseRandomNumberSeed)

experiment.performSimulation()

