### VERSION "nCore"
## ../nCore/bin/ClassifierUsingAutocoderDataSetBP.rb
# Purpose:  Simple 4-class classifier of data used in 4-cluster AutocoderBP.rb.

require_relative 'BaseLearningExperiment'

class Neuron
  include NonMonotonicIOFunction
  # include PiecewiseLinNonMonIOFunction
end

class OutputNeuron
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
        :probabilityOfBeingEnabled => 0.5,
        :outputLayerLearningRate => 0.1,
        :hiddenLayerLearningRate => 0.1,
        :outputErrorLearningRate => nil,
        :minMSE => 0.0, # 0.001,
        :maxNumEpochs => 7e3,

        # Network Architecture
        :numberOfInputNeurons => 3,
        :numberOfHiddenNeurons => 2,
        :numberOfOutputNeurons => 4,
        :weightRange => 1.0,
        :typeOfLink => Link,
        :typeOfNeuron => Neuron, # NoisyNeuron,
        :typeOfOutputNeuron => OutputNeuron,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 16),
        :numberOfTestingExamples => numberOfExamples,

        # Recording and database parameters
        :neuronsToDisplay => [8],
        :intervalForSavingNeuronData => 100000,
        :intervalForSavingDetailedNeuronData => 2000,
        :intervalForSavingTrainingData => 100
    }
  end

  def createTrainingSet
    xStart = [-1.0, 1.0, -1.0, 1.0]
    yStart = [1.0, 1.0, -4.0, -4.0]
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
        aPoint = [x, y, z]
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
    network = Standard3LayerNetwork.new(args)
    theTrainer = StandardBPTrainingSupervisorModLR.new(examples, network, args)
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

    plotMSEvsEpochNumber(trainingDataRecords) unless (trainingDataRecords.nil?)
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("3in1hidN4ClAuto   Classifier Using Autocoder DataSet with hidden NonMonUnShifted", baseRandomNumberSeed)

experiment.performSimulation()