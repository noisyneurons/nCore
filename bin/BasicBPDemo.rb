### VERSION "nCore"
## ../nCore/bin/BasicBPDemo.rb
## Simple backprop demo. For XOR, and given parameters, requires 2080 epochs to converge.
##                       For OR, and given parameters, requires 166 epochs to converge.

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralParts2'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers.rb'

require_relative 'BaseLearningExperiment'


class Experiment

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters re. Output Error
        :learningRate => 3.0,
        :minMSE => 0.001,
        :trainingSequence =>  TrainingSequence,
        :maxNumEpochs => 2e3,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 2,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,

        :typeOfLink => Link,
        :typeOfNeuron => Neuron2,
        :typeOfOutputNeuron => OutputNeuron2,
     #   :typeOfLinkToOutput => Link,


        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 4),
        :numberOfTestingExamples => numberOfExamples,

    }
  end

  def createDataSet
    examples = []
    examples << {:inputs => [0.0, 0.0], :targets => [0.0], :exampleNumber => 0, :class => 0}
    examples << {:inputs => [0.0, 1.0], :targets => [1.0], :exampleNumber => 1, :class => 1}
    examples << {:inputs => [1.0, 0.0], :targets => [1.0], :exampleNumber => 2, :class => 1}
    examples << {:inputs => [1.0, 1.0], :targets => [0.0], :exampleNumber => 3, :class => 0}
    return examples
  end

  def createNetworkAndTrainer
    network = Standard3LayerNetwork.new(args)
    theTrainer = TrainerBase.new(examples, network, args)
    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("BasicBPDemo1 both layers NonMon", baseRandomNumberSeed)

experiment.performSimulation()