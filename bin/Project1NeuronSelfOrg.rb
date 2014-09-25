### VERSION "nCore"
## ../nCore/bin/Project1NeuronSelfOrg.rb
# Purpose:  1NeuronSelfOrg;  Get simplest versions of self-org understood and "working."

require_relative 'BaseLearningExperiment'

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralParts2'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers'
require_relative '../lib/core/NeuralSelfOrg'

require_relative 'BaseLearningExperiment'


class Neuron2
  include NonMonotonicIOFunction
end

class Experiment

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => (randomNumberSeed + 0),

        # training parameters
        :learningRate => 0.1,
        :minMSE => 0.0,
        :maxEpochNumbersForEachPhase => [1, 200, 200, 6e2, 200, 6e2, 200],
        :trainingSequence => MultiPhaseTrainingSequence,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenLayer1Neurons => 1,
        #:numberOfOutputNeurons => 1,
        :weightRange => 0.1,
        :typeOfLink => LinkWithNormalization,
        :typeOfNeuron => Neuron2,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 16),
        :numberOfTestingExamples => numberOfExamples
    }
  end

  def createDataSet
    # xStart = [-1.0, 1.0, -1.0, 1.0]
    # xStart = [0.0, 2.0, 0.0, 2.0]
    xStart = [1.0, 3.0, 1.0, 3.0]

    yStart = [1.0, 1.0, -1.0, -1.0]
    # yStart = [2.0, 2.0, 0.0, 0.0]
    xInc = [0.0, 0.0, 0.0, 0.0]
    # yInc = [0.2, 0.2, -0.2, -0.2]
    yInc = [0.0, 0.0, -0.0, -0.0]

    numberOfClasses = xStart.length
    numberOfExamplesInEachClass = numberOfExamples / numberOfClasses
    exampleNumber = 0
    examples = []
    numberOfClasses.times do |indexToClass|
      xS = xStart[indexToClass]
      xI = xInc[indexToClass]
      yS = yStart[indexToClass]
      yI = yInc[indexToClass]

      numberOfExamplesInEachClass.times do |classExNumb|
        x = xS + (xI * classExNumb)
        y = yS + (yI * classExNumb)
        aPoint = [x, y]
        desiredOutputs = [0.0, 0.0, 0.0, 0.0]
        desiredOutputs[indexToClass] = 1.0
        examples << {:inputs => aPoint, :targets => desiredOutputs, :exampleNumber => exampleNumber, :class => indexToClass}
        exampleNumber += 1
      end
    end
    STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
    puts examples
    examples
  end


  def createNetworkAndTrainer
    network = SelfOrg1NeuronNetwork.new(args)

    selfOrgLayer = network.allNeuronLayers[1]
    selfOrgNeuron = selfOrgLayer[0]
    selfOrgNeuron.inputLinks[0].weight = 0.1
    selfOrgNeuron.inputLinks[1].weight = 0.05
    # selfOrgNeuron.inputLinks[2].weight = 100.0

    theTrainer = TrainerSelfOrgWithLinkNormalization.new(examples, network, args)

    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("Proj1SelfOrg: single self-org neuron", baseRandomNumberSeed)

experiment.performSimulation()