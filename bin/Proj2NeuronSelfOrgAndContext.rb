### VERSION "nCore"
## ../nCore/bin/Proj2NeuronSelfOrgAndContext.rb

# Specific Purpose for this experiment: Get simplest versions of self-org AND CONTEXT understood and "working."
# General Purpose:  Start of Project 7; project to split example set to learn sub-parts, and then combine those parts/neuron-functions that
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
        :learningRate =>  0.1,
        :minMSE => 0.0, # 0.001,
        :maxEpochNumbersForEachPhase => [1, 150, 1, 150],
        :trainingSequence =>  MultiPhaseTrainingSequence,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenLayer1Neurons => 1,
        :numberOfHiddenLayer2Neurons => 2,
        :numberOfOutputNeurons => 4,


        :weightRange => 0.1,

        :typeOfLink => LinkWithNormalization,
        :typeOfNeuron => Neuron2,
        :typeOfLinkToOutput => Link,
        :typeOfOutputNeuron => OutputNeuron2,

        # Training Set parameters

        :numberOfExamples => (self.numberOfExamples = 16),
        :numberOfTestingExamples => numberOfExamples,
        :standardDeviationOfAddedGaussianNoise => 1e-24,
        :angleOfClockwiseRotationOfInputData => 0.0
    }
  end


  #### generator to try different simple configurations:  e.g. where one of the input axises contains more "noise" than the other axises.
  def genTemp
    gaussianRandomNumberGenerator = NormalDistribution.new(meanOfGaussianNoise = 0.0,  args[:standardDeviationOfAddedGaussianNoise])

    xStart = [-1.0, 1.0, -1.0, 1.0]
    yStart = [1.0, 1.0, -1.0, -1.0]


    xIncVal = 0.0 # 0.002   # "noise level"
    xInc = [-xIncVal, xIncVal, -xIncVal, xIncVal]

    yIncVal = 0.0      # "noise level"
    yInc = [yIncVal, yIncVal, -yIncVal, -yIncVal]

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
        x = xS + (xI * classExNumb) + gaussianRandomNumberGenerator.get_rng
        y = yS + (yI * classExNumb) + gaussianRandomNumberGenerator.get_rng
        aPoint = [x, y]
        desiredOutputs = [0.0, 0.0, 0.0, 0.0]
        desiredOutputs[indexToClass] = 1.0
        examples << {:inputs => aPoint, :targets => desiredOutputs, :exampleNumber => exampleNumber, :class => indexToClass}
        exampleNumber += 1
      end
    end
    STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
    angleOfClockwiseRotationOfInputData = args[:angleOfClockwiseRotationOfInputData]
    examples = rotateClockwise(examples, angleOfClockwiseRotationOfInputData)
  end

  def createDataSet
    genTemp # gen4ClassDS
  end

  #def temporarilySetSpecificWeights(network)
  #  selfOrgLayer = network.allNeuronLayers[1]
  #  selfOrgNeuron = selfOrgLayer[0]
  #  selfOrgNeuron.inputLinks[0].weight = 0.1
  #  selfOrgNeuron.inputLinks[1].weight = 0.1
  #end

  def createNetworkAndTrainer
    network = Context4LayerNetwork.new(args)

    # temporarilySetSpecificWeights(network)

    theTrainer = Trainer2SelfOrgAndContext.new(examples, network, args)

    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("Proj2NeuronSelfOrgAndContext; 2 in 4 out; divide then integrate", baseRandomNumberSeed)

experiment.performSimulation()

puts experiment.network