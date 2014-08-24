### VERSION "nCore"
## ../nCore/bin/BasicSelfOrgDemo.rb
## Simple backprop demo. For XOR, and given parameters, requires 2080 epochs to converge.
##                       For OR, and given parameters, requires 166 epochs to converge.

require_relative 'BaseLearningExperiment'

class Neuron
  include NonMonotonicIOFunction
  include SelfOrganization
end

class OutputNeuron
  #include NonMonotonicIOFunction
 #include LinearIOFunction
end


class Experiment
  include NeuronToNeuronConnection

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters re. Output Error
        :learningRate => 0.2,
        :minMSE => 10e-5,
        :maxEpochNumbersForEachPhase => [600,600],
        :trainingSequence =>  MultiPhaseTrainingSequence,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 1,
        :numberOfOutputNeurons => 1,
        :weightRange => 0.01,

        :typeOfLink => Link,
        :typeOfNeuron => Neuron,
        :typeOfOutputNeuron => OutputNeuron,
        #   :typeOfLinkToOutput => Link,


        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 4),
        :numberOfTestingExamples => numberOfExamples,


        # Recording and database parameters
        :neuronsToDisplay => [5],
        :intervalForSavingNeuronData => 100,
        :intervalForSavingDetailedNeuronData => 1000,
        :intervalForSavingTrainingData => 100
    }
  end

  def createTrainingSet
    examples = []
    examples << {:inputs => [0.0, 0.0], :targets => [0.0], :exampleNumber => 0, :class => 0}
    examples << {:inputs => [0.0, 1.0], :targets => [0.0], :exampleNumber => 1, :class => 1}
    examples << {:inputs => [1.0, 0.0], :targets => [1.0], :exampleNumber => 2, :class => 1}
    examples << {:inputs => [1.0, 1.0], :targets => [1.0], :exampleNumber => 3, :class => 0}
    return examples
  end

  def createTestingSet
    return createTrainingSet
  end

  def createNetworkAndTrainer
    network = Standard3LayerNetwork.new(args)
    theTrainer = TrainerSelfOrg.new(examples, network, args)
    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("BasicBPDemo1 both layers NonMon", baseRandomNumberSeed)

experiment.performSimulation()