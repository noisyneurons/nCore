### VERSION "nCore"
## ../nCore/bin/Project7pt1.rb
# Purpose:  Start of Project 7; project to split example set to learn sub-parts, and then combine those parts/neuron-functions that
# didn't need to be separated, but instead need to be integrated to obtain better generalization.
# Ultimate goal of project 6 is develop analogy processing -- where one function useful for solving one problem
# can be of use in solving another problem.  The common function(s)/neuron(s) can be thus be 'reused' -- and even potentially made
# better by improving the accuracy of the function parameters because more examples are used to learn the parameters.

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralContext'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers.rb'

require_relative 'BaseLearningExperiment'

class Neuron
  include NonMonotonicIOFunction
  include SelfOrganization
end

class OutputNeuron
end


class Experiment
  include NeuronToNeuronConnection

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters
        :learningRate => 0.1,
        :minMSE => 0.0, # 0.001,
        :maxEpochNumbersForEachPhase => [200, 6e2, 200, 6e2, 200, 6e2, 200],
        :trainingSequence => MultiPhaseTrainingSequence,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenLayer1Neurons => 1,
        :numberOfHiddenLayer2Neurons => 2,
        :numberOfOutputNeurons => 4,
        :weightRange => 0.1,
        :typeOfLink => LinkInContext,
        :typeOfLinkToOutput => Link,
        :typeOfNeuron => NeuronInContext,
        :typeOfOutputNeuron => OutputNeuron,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 16),
        :numberOfTestingExamples => numberOfExamples,

        # Recording and database parameters
        :neuronsToDisplay => [6],
        :intervalForSavingNeuronData => 100, #100000,
        :intervalForSavingDetailedNeuronData => 100, #2000,
        :intervalForSavingTrainingData => 100
    }
  end

  def createTrainingSet
    xStart = [-1.0, 1.0, -1.0, 1.0]
    yStart = [1.0, 1.0, -1.0, -1.0]
    xInc = [0.0, 0.0, 0.0, 0.0]
    yInc = [0.2, 0.2, -0.2, -0.2]

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
    logger.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
    logger.puts examples
    examples
  end

  def createTestingSet
    return createTrainingSet
  end

  def createNetworkAndTrainer
    network = Context4LayerNetworkVer1.new(args)
    theTrainer = Trainer7pt1.new(examples, network, args)
    return network, theTrainer
  end

end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("Proj7pt1; 2 in 4 out; divide then integrate", baseRandomNumberSeed)

experiment.performSimulation()