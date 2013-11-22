### VERSION "nCore"
## ../nCore/bin/BasicBPDemoSharedWeight.rb
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
        :numberOfHiddenNeurons => 6,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => (numExamples = 4),
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
    examples << {:inputs => [0.0, 0.0], :targets => [0.1], :exampleNumber => 0, :class => 0}
    examples << {:inputs => [0.0, 1.0], :targets => [0.9], :exampleNumber => 1, :class => 1}
    examples << {:inputs => [1.0, 0.0], :targets => [0.9], :exampleNumber => 2, :class => 1}
    examples << {:inputs => [1.0, 1.0], :targets => [0.1], :exampleNumber => 3, :class => 0}
    return examples
  end

  def createNetworkAndTrainer
    network = Flocking3LayerNetwork.new(args) # we rally don't use the flocking part of the network here! -- but we need...

    sendingLayer = network.inputLayer
    receivingLayer = hiddenLayer = network.allNeuronLayers[1]
    numberOfGroups = 2
    shareWeightsBetweenNGroups(sendingLayer, receivingLayer, numberOfGroups)

    theTrainer = StandardBPTrainingSupervisor.new(examples, network, args)
    return network, theTrainer
  end

end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("SharedWBPD1", baseRandomNumberSeed)

experiment.performSimulation()

