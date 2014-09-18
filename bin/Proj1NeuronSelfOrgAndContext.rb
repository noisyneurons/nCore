### VERSION "nCore"
## ../nCore/bin/Proj1NeuronSelfOrgAndContext.rb
# Purpose: Get simplest versions of self-org AND CONTEXT understood and "working."

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
# require_relative '../lib/core/NeuralContext'
# require_relative '../lib/core/NeuralSelfOrg'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers.rb'
require_relative '../lib/core/NeuralSelfOrg'
require_relative '../lib/core/NeuralContext'

require_relative 'BaseLearningExperiment'


########################################################################
class NeuronInContext
  include NonMonotonicIOFunction
  include SelfOrganization
end

class LinkWithNormalizationAndContext < LinkWithNormalization
  include LearningInContext
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
        :maxEpochNumbersForEachPhase => [200, 6e2, 200, 6e2, 200, 6e2, 200],
        :trainingSequence =>  MultiPhaseTrainingSequence,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenLayer1Neurons => 1,
        :numberOfOutputNeurons => 1,
        :weightRange => 0.1,
        #:typeOfLink => Link,
        :typeOfLink => LinkWithNormalizationAndContext,
        #:typeOfLinkToOutput => Link,
        :typeOfNeuron => NeuronInContext,
        :typeOfOutputNeuron => OutputNeuron,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 16),
        :numberOfTestingExamples => numberOfExamples,

        # Recording and database parameters
        :neuronsToDisplay => [1],
        :intervalForSavingNeuronData => 100, #100000,
        :intervalForSavingDetailedNeuronData => 100, #2000,
        :intervalForSavingTrainingData => 100
    }
  end

  def createTrainingSet
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

  def createTestingSet
    return createTrainingSet
  end

  def createNetworkAndTrainer
    network = SelfOrg1NeuronNetwork.new(args)

    selfOrgLayer = network.allNeuronLayers[1]
    selfOrgNeuron = selfOrgLayer[0]
    selfOrgNeuron.inputLinks[0].weight = 0.1
    selfOrgNeuron.inputLinks[1].weight = 0.01
    # selfOrgNeuron.inputLinks[2].weight = 100.0

    theTrainer = TrainerSelfOrgWithLinkNormalization.new(examples, network, args)

    return network, theTrainer
  end

  def reportTrainingResults(neuronToDisplay, descriptionOfExperiment, lastEpoch, lastTrainingMSE, lastTestingMSE, network, startingTime)

    endOfTrainingReport(lastEpoch, lastTestingMSE, lastTrainingMSE, network)

    #neuronDataSummary(neuronToDisplay)

    #detailedNeuronDataSummary(neuronToDisplay)

    trainingDataRecords = trainingDataSummary

    storeSnapShotData(descriptionOfExperiment, lastEpoch, lastTestingMSE, lastTrainingMSE, network, startingTime)

    snapShotDataSummary

    #plotMSEvsEpochNumber(trainingDataRecords)

    # plotTrainingResults(neuronToDisplay)
  end


end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("Proj1NeuronSelfOrgAndContext; 2 in 4 out; divide then integrate", baseRandomNumberSeed)

experiment.performSimulation()