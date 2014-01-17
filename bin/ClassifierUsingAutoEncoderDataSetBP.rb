### VERSION "nCore"
## ../nCore/bin/ClassifierUsingAutocoderDataSetBP.rb
# Purpose:  Simple 4-class classifier of data used in 4-cluster AutocoderBP.rb.

require_relative 'BaseLearningExperiment'

class Neuron
  include NonMonotonicIOFunction
end

class OutputNeuron
  include NonMonotonicIOFunction
end


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
        :maxNumEpochs => 1100, # 6e3,

        # Network Architecture
        :numberOfInputNeurons => 3,
        :numberOfHiddenNeurons => 2,
        :numberOfOutputNeurons => 4,
        :weightRange => 1.0,
        :typeOfLink => Link,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 16),

        # Recording and database parameters
        :neuronToDisplay => 5,
        :intervalForSavingNeuronData => 100,
        :intervalForSavingDetailedNeuronData => 1000,
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
    STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
    puts examples
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

experiment = Experiment.new("3in1hidN4ClAuto   Classifier Using Autocoder DataSet with hidden NonMonUnShifted", baseRandomNumberSeed)

experiment.performSimulation()