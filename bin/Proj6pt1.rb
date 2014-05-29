### VERSION "nCore"
## ../nCore/bin/Proj6pt1.rb
# Purpose:  Start of Project 6; single non-mon output neuron
# Ultimate goal of project 6 is to preliminarily test non-mon neurons' flocking and self-org to create features.

require_relative 'BaseLearningExperiment'

#class Neuron
#  include NonMonotonicIOFunction
#  # include PiecewiseLinNonMonIOFunction
#end

class OutputNeuron
  # include SigmoidIOFunction
  # include NonMonotonicIOFunction
  # include PiecewiseLinNonMonIOFunction
end


class Experiment
  include NeuronToNeuronConnection

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters
        :outputErrorLearningRate => 0.1,
        :minMSE => 0.0, # 0.001,
        :maxNumEpochs => 7e3,
        # :probabilityOfBeingEnabled => 0.5,


        # Network Architecture
        :numberOfInputNeurons => 2,
        # :numberOfHiddenNeurons => 0,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => Link,
        # :typeOfNeuron => Neuron, # NoisyNeuron,
        :typeOfOutputNeuron => OutputNeuron,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 16),
        :numberOfTestingExamples => numberOfExamples,

        # Recording and database parameters
        :neuronsToDisplay => [0],
        :intervalForSavingNeuronData => 100000,
        :intervalForSavingDetailedNeuronData => 2000,
        :intervalForSavingTrainingData => 100
    }
  end

  def createTrainingSet
    xStart = [-1.0, 1.0, -1.0, 1.0]
    yStart = [1.0, 1.0, -1.0, -1.0]
    xInc = [0.0, 0.0, 0.0, 0.0]
    yInc = [1.0, 1.0, 1.0, 1.0]
    zS = -2.0 # 0.0 # -2.0
    zI = 1.0 # 0.0 # 1.0


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
        z = zS + (zI * classExNumb)
        # aPoint = [x, y, z]
        aPoint = [x, y]
        desiredOutputs = [0.0]
        desiredOutputs[0] = 1.0  if(indexToClass == 1)
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
    network = Simplest1LayerNet.new(args)
    theTrainer = BPTrainingSupervisorFor1LayerNet.new(examples, network, args)
    return network, theTrainer
  end

  def reportTrainingResults(neuronsToDisplay, descriptionOfExperiment, lastEpoch, lastTrainingMSE, lastTestingMSE, network, startingTime)
    trainingDataRecords = nil

    endOfTrainingReport(lastEpoch, lastTestingMSE, lastTrainingMSE, network)

    neuronDataSummary(neuronsToDisplay)

    detailedNeuronDataSummary(neuronsToDisplay)

    trainingDataRecords = trainingDataSummary

    storeSnapShotData(descriptionOfExperiment, lastEpoch, lastTestingMSE, lastTrainingMSE, network, startingTime)

    snapShotDataSummary

    plotMSEvsEpochNumber(trainingDataRecords) unless(trainingDataRecords.nil?)
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("Proj6pt1; 2 in 1 out; single NonMon Output Neuron; No Hidden Layer", baseRandomNumberSeed)

experiment.performSimulation()