### VERSION "nCore"
## ../nCore/bin/CircleBP.rb

require_relative 'BaseLearningExperiment'

class Neuron
  include NonMonotonicIOFunctionUnShifted
end

#class OutputNeuron
#  include NonMonotonicIOFunctionUnShifted
#end


class Experiment

  def setParameters

    args = {
        :experimentNumber => experimentLogger.experimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.3, # 0.03,
        :minMSE => 0.001,
        :maxNumEpochs => 12e3,

        # Network Architecture and initial weights
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 20,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => Link,

        # Training Set parameters
        :numberOfExamples => (numberOfExamples = 24),
        :firstExamplesAngleToXAxis => 0.0,
        :firstExamplesAngleToXAxisForTesting => 15.0,

        # Recording and database parameters
        :neuronToDisplay => 2,
        :intervalForSavingNeuronData => 10000,
        :intervalForSavingDetailedNeuronData => 10000,
        :intervalForSavingTrainingData => 100,
    }
  end

  def createTrainingSet
    firstExamplesAngleToXAxis = args[:firstExamplesAngleToXAxis]
    numberOfExamples = args[:numberOfExamples]
    examples = createExamples(firstExamplesAngleToXAxis, numberOfExamples)
    return examples
  end

  def createTestingSet
    firstExamplesAngleToXAxisForTesting = args[:firstExamplesAngleToXAxisForTesting]
    numberOfExamples = args[:numberOfExamples]
    testingExamples = createExamples(firstExamplesAngleToXAxisForTesting, numberOfExamples)
    args[:testingExamples] = testingExamples
  end

  def createExamples(firstExamplesAngleToXAxis, numberOfExamples)
    numberOfClasses = 2
    numExamplesPerClass = numberOfExamples / numberOfClasses

    angleBetweenExamplesInDegrees = 360.0 / numExamplesPerClass
    radiusArray = [1.0, 1.3]
    desiredOutput = [0.0, 1.0]

    examples = []
    numExamplesPerClass.times do |exampleNumberWithinClass|
      angle = (angleBetweenExamplesInDegrees * exampleNumberWithinClass) + firstExamplesAngleToXAxis
      angleInRadians = angle * (360.0/(2.0 * Math::PI))
      numberOfClasses.times do |indexToClass|
        radius = radiusArray[indexToClass]
        x = radius * Math.cos(angleInRadians)
        y = radius * Math.sin(angleInRadians)
        aPoint = [x, y]
        targets = [desiredOutput[indexToClass]]
        exampleNumber = if (indexToClass == 1)
                          exampleNumberWithinClass + numExamplesPerClass
                        else
                          exampleNumberWithinClass
                        end
        anExample = {:inputs => aPoint, :targets => targets, :exampleNumber => exampleNumber, :class => indexToClass}
        examples << anExample
      end
    end
    STDERR.puts "Error:   Incorrect Number of Examples Generated and/or Specified" unless (examples.length == args[:numberOfExamples])
    examples
  end

  def createNetworkAndTrainer
    network = Standard3LayerNetwork.new(args)
    theTrainer = StandardBPTrainingSupervisor.new(examples, network, args)
    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

experiment = Experiment.new("NonMonL1CircleBP the output layer has sigmoid IO but the output layer has sigmoid", randomNumberSeed=0)

experiment.performSimulation()
