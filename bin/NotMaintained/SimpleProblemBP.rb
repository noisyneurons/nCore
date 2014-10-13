### VERSION "nCore"
## ../nCore/bin/SimpleProblemBP.rb

require_relative 'BaseLearningExperiment'


class OutputNeuron
  # include NonMonotonicIOFunctionUnShifted
  include NonMonotonicIODerivative
end


class Experiment
  include NeuronToNeuronConnection

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.1,
        :minMSE => 0.0000001,
        :maxNumEpochs => 4e3,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => Link,

        # Training Set parameters
        :numberOfExamples => (numExamples = 8),
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
    examples << {:inputs => [1.0, 1.0], :targets => [1.0], :exampleNumber => 0, :class => 1}
    examples << {:inputs => [1.0, 2.0], :targets => [1.0], :exampleNumber => 1, :class => 1}
    examples << {:inputs => [1.0, 3.0], :targets => [1.0], :exampleNumber => 2, :class => 1}
    examples << {:inputs => [1.0, 4.0], :targets => [1.0], :exampleNumber => 3, :class => 1}
    examples << {:inputs => [-1.0, -1.0], :targets => [0.0], :exampleNumber => 4, :class => 0}
    examples << {:inputs => [-1.0, -2.0], :targets => [0.0], :exampleNumber => 5, :class => 0}
    examples << {:inputs => [-1.0, -3.0], :targets => [0.0], :exampleNumber => 6, :class => 0}
    examples << {:inputs => [-1.0, -4.0], :targets => [0.0], :exampleNumber => 7, :class => 0}
    examples
  end

  def createNetworkAndTrainer

    network = Simplest1LayerNet.new(args) # we rally don't use the flocking part of the network here! -- but we need...
    theTrainer = BPTrainingSupervisorFor1LayerNet.new(examples, network, args)

    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("NonMonIO1LayerSimplestProblemBP  NonMonotonicIOFunctionUnShifted IO function in 1 layer net on simplest 2-D problem", baseRandomNumberSeed)

experiment.performSimulation()

