### VERSION "nCore"
## ../nCore/bin/ProjAD1.rb

# Specific Purpose for this experiment: Implement and test simple A to D converter learning...
# Purpose:  Start of Project 7; project to split example set to learn sub-parts, and then combine those parts/neuron-functions that
# didn't need to be separated, but instead need to be integrated to obtain better generalization.
# Ultimate goal of project 6 is develop analogy processing -- where one function useful for solving one problem
# can be of use in solving another problem.  The common function(s)/neuron(s) can be thus be 'reused' -- and even potentially made
# better by improving the accuracy of the function parameters because more examples are used to learn the parameters.

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralParts2'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/core/NetworkFactories2'
require_relative '../lib/core/Trainers'
require_relative '../lib/core/Trainers2'

require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'

require_relative 'BaseLearningExperiment'
########################################################################

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
        :minMSE => 0.0, # 0.001,
        :maxEpochNumbersForEachPhase => [1, 1150, 1, 1150, 4000],
        :trainingSequence => MultiPhaseTrainingSequence,

        # Network Architecture
        :numberOfInputNeurons => 1,
        :numberOfHiddenLayer1Neurons => 1,
        :numberOfHiddenLayer2Neurons => 2,
        :numberOfOutputNeurons => 2,

        :weightRange => 0.1,

        :typeOfLink => LinkWithNormalization,
        :typeOfNeuron => Neuron2,
        :typeOfLinkToOutput => Link,
        :typeOfOutputNeuron => OutputNeuron2,

        # Training Set parameters
        :angleOfClockwiseRotationOfInputData => 0.0,
        :standardDeviationOfAddedGaussianNoise => 1e-24,
        :numberOfExamples => (self.numberOfExamples = 20),
        :numberOfTestingExamples => numberOfExamples,

    }
  end

  def createDataSet
    xStart = [-1.0, -0.5, 0.0, 0.5]
    xInc = [0.1, 0.1, 0.1, 0.1]

    numberOfClasses = xStart.length
    numberOfExamplesInEachClass = numberOfExamples / numberOfClasses
    exampleNumber = 0
    examples = []
    numberOfClasses.times do |indexToClass|
      xS = xStart[indexToClass]
      xI = xInc[indexToClass]

      numberOfExamplesInEachClass.times do |classExNumb|
        x = xS + (xI * classExNumb)
        aPoint = [x]

        msb = 0
        msb = 1 if x >= 0.0
        if (msb == 1)
          lsb = 0
          lsb = 1 if x >= 0.5
        else
          lsb = 0
          lsb = 1 if x >= -0.5
        end
        desiredOutputs = [msb, lsb]

        examples << {:inputs => aPoint, :targets => desiredOutputs, :exampleNumber => exampleNumber, :class => indexToClass}
        exampleNumber += 1
      end
    end
    logger.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
    examples
  end

  def createNetworkAndTrainer
    network = Context4LayerNetwork.new(args)

    selfOrgLayer = network.allNeuronLayers[1]
    selfOrgNeuron = selfOrgLayer[0]
    selfOrgNeuron.inputLinks[0].weight = 0.105
    selfOrgNeuron.inputLinks[1].weight = 0.1

    theTrainer = TrainerAD1.new(examples, network, args)

    return network, theTrainer
  end
end

###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("ProjAD1; 1 in 2 out; A to D converter problem", baseRandomNumberSeed)

experiment.performSimulation()

logger.puts experiment.network