### VERSION "nCore"
### VERSION "nCore"
## ../nCore/bin/Proj6pt2Variations.rb
# Purpose:  Second and very important part of Project 6;
# 3-layer standard network, except that output neuron is non-mon


require_relative 'BaseLearningExperiment'

class Neuron
  # include LinearIOFunction
  # include NonMonotonicIOFunction
  # include PiecewiseLinNonMonIOFunction
end

class OutputNeuron
  include LinearIOFunction
  # include SigmoidIOFunction
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
        :outputErrorLearningRate => 0.001,
        :minMSE => 0.0, # 0.001,
        :maxNumEpochs => 30e3,
        :probabilityOfBeingEnabled => 1.0,


        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 2,
        :numberOfOutputNeurons => 2,
        :weightRange => 0.01,
        :typeOfLink => Link,
        :typeOfNeuron => Neuron,
        :typeOfOutputNeuron => OutputNeuron,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 16),
        :numberOfTestingExamples => numberOfExamples,

        # Recording and database parameters
        :neuronsToDisplay => [5],
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
    # yInc = [0.0, 0.0, 0.0, 0.0]
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

        desiredOutputs = aPoint

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
    theTrainer = StandardBPTrainingSupervisor.new(examples, network, args)
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

experiment = Experiment.new("Proj6pt2; 2-in 1-out; single NonMon Output Neuron; 2-4 Neurons in Hidden Layer", baseRandomNumberSeed)

experiment.performSimulation()