### VERSION "nCore"
## ../nCore/bin/ClassifierUsingAutocoderDataSetBP.rb
# Purpose:  Simple backprop autocoder implementation.

require_relative 'BaseLearningExperiment'

class Neuron
  include NonMonotonicIOFunctionUnShifted
end

#class OutputNeuron
#  include NonMonotonicIOFunctionUnShifted
#end


class Experiment
  include NeuronToNeuronConnection

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters re. Output Error
        :outputLayerLearningRate => 0.1,
        :hiddenLayerLearningRate => 0.1,
        :outputErrorLearningRate => nil,
        :minMSE => 0.00001, # 0.001,
        :maxNumEpochs => 3e3,  # 6e3,

        # Network Architecture
        :numberOfInputNeurons => 3,
        :numberOfHiddenNeurons => 2,
        :numberOfOutputNeurons => 4,
        :weightRange => 1.0,
        :typeOfLink => Link,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 16),

        # Recording and database parameters
        :neuronToDisplay => 2,
        :intervalForSavingNeuronData => 100,
        :intervalForSavingDetailedNeuronData => 1000,
        :intervalForSavingTrainingData => 100
    }
  end

  def createTrainingSet
    createTrainingSet3Inputs
  end


  #def createTrainingSet2Inputs
  #  xStart = [-1.0, 1.0, -1.0, 1.0]
  #  yStart = [1.0, 1.0, -1.0, -1.0]
  #  xInc = [0.0, 0.0, 0.0, 0.0]
  #  yInc = [1.0, 1.0, -1.0, -1.0]
  #
  #  numberOfClasses = xStart.length
  #  numberOfExamplesInEachClass = numberOfExamples / numberOfClasses
  #  exampleNumber = 0
  #  xIncrement = 0.0
  #  yIncrement = 0.0
  #  examples = []
  #  numberOfClasses.times do |indexToClass|
  #    xS = xStart[indexToClass]
  #    xI = xInc[indexToClass]
  #    yS = yStart[indexToClass]
  #    yI = yInc[indexToClass]
  #    numberOfExamplesInEachClass.times do |classExNumb|
  #      x = xS + (xI * classExNumb)
  #      y = yS + (yI * classExNumb)
  #      aPoint = [x, y]
  #      desiredOutputs = [0.0, 0.0, 0.0, 0.0]
  #      desiredOutputs[indexToClass] = 1.0
  #      examples << {:inputs => aPoint, :targets => desiredOutputs, :exampleNumber => exampleNumber, :class => indexToClass}
  #      exampleNumber += 1
  #    end
  #  end
  #  STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  #  examples
  #end


  def createTrainingSet3Inputs
    xStart = [-1.0, 1.0, -1.0, 1.0]
    yStart = [1.0, 1.0, -4.0, -4.0]
    xInc = [0.0, 0.0, 0.0, 0.0]
    yInc = [1.0, 1.0, 1.0, 1.0]
    zS = -2.0
    zI = 1.0


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
    STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
    examples
  end


  def createNetworkAndTrainer
    network = Standard3LayerNetwork.new(args)
    theTrainer = StandardBPTrainingSupervisorModLR.new(examples, network, args)
    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("3DNonMonUnShiftedClAutoEnc   Classifier Using Autocoder DataSet with hidden NonMonUnShifted", baseRandomNumberSeed)

experiment.performSimulation()