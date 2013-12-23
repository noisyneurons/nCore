### VERSION "nCore"
## ../nCore/bin/AutocoderBP.rb
# Purpose:  Simple backprop autocoder implementation.

require_relative 'BaseLearningExperiment'

#class Neuron
#  include NonMonotonicIOFunction
#end

class OutputNeuron
  include LinearIOFunction
end



class Experiment
  include NeuronToNeuronConnection

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.001,
        :minMSE => 0.001,
        :maxNumEpochs => 6e3,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 8,
        :numberOfOutputNeurons => 2,
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

    xStart = [-1.0, 1.0, -1.0, 1.0]
    yStart = [1.0, 1.0, -1.0, -1.0]
    xInc = [0.0, 0.0, 0.0, 0.0]
    yInc = [1.0, 1.0, -1.0, -1.0]

    numberOfClasses = xStart.length
    numberOfExamplesInEachClass = numberOfExamples / numberOfClasses
    exampleNumber = 0
    xIncrement = 0.0
    yIncrement = 0.0
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
        examples << {:inputs => aPoint, :targets => aPoint, :exampleNumber => exampleNumber, :class => indexToClass}
        exampleNumber += 1
      end
    end
    STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
    examples
  end


  def createNetworkAndTrainer
    network = Standard3LayerNetwork.new(args)
    theTrainer = StandardBPTrainingSupervisor.new(examples, network, args)
    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("SigAutoEncoder output neuron linear", baseRandomNumberSeed)

experiment.performSimulation()