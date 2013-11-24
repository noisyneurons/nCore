### VERSION "nCore"
## ../nCore/bin/ThreeClass2HiddenLayers.rb
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
        :minMSE => 0.001,
        :maxNumEpochs => 4e3,

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
        :neuronToDisplay => 2,
        :intervalForSavingNeuronData => 100,
        :intervalForSavingDetailedNeuronData => 1000,
        :intervalForSavingTrainingData => 100
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
    network = Recurrent2HiddenLayerNetworkSpecial.new(args) # we rally don't use the flocking part of the network here! -- but we need...
    puts network.to_s
    theTrainer = StandardBPTrainingSupervisor.new(examples, network, args)
    return network, theTrainer
  end

end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("3Cls2Hid2Shared", baseRandomNumberSeed)

experiment.createNetworkAndTrainer

#experiment.performSimulation()

