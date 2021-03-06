### VERSION "nCore"
## ../nCore/bin/CircleBPofFlockErrorSciNet.rb

require_relative 'BaseLearningExperiment'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

class Experiment

  def setParameters

    args = {
        :experimentNumber => experimentLogger.experimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.3, # 0.03,
        :minMSE => 0.001,
        :maxNumEpochs => 2e3, # 120e3

        # Network Architecture and initial weights
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 20,
        :numberOfOutputNeurons => 3,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => (numberOfExamples = 24),
        :firstExamplesAngleToXAxis => 0.0,
        :firstExamplesAngleToXAxisForTesting => 15.0,

        # Recording and database parameters
        :intervalForSavingNeuronData => 10000,
        :intervalForSavingDetailedNeuronData => 10000,
        :intervalForSavingTrainingData => 1000,

        # Flocking Parameters...
        :flockingLearningRate => -0.009, # -0.0002,
        :bpFlockingLearningRate => -0.063,
        :maxFlockingIterationsCount => 60, # 2000,
        :targetFlockIterationsCount => 20,
        :ratioDropInMSE => 0.95,
        :ratioDropInMSEForFlocking => 0.96,
        # :maxAbsFlockingErrorsPerExample => 0.2, #  0.04 / numberOfExamples = 0.005
        :ratioDropInMSEForBPFlocking => 0.98,
        :maxBPFlockingIterationsCount => 60,
        :targetBPFlockIterationsCount => 20,


        # Flocker Specs...
        :typeOfClusterer => DynamicClusterer,
        :numberOfClusters => 2,
        :m => 2.0,
        :numExamples => numberOfExamples,
        :exampleVectorLength => 1,
        :delta => 1e-2,
        :maxNumberOfClusteringIterations => 10,
        :keepTargetsSymmetrical => true,
        :targetDivergenceFactor => 1.0,
        :alwaysUseFuzzyClusters => true,
        :epochsBeforeFlockingAllowed => 0, #  10e1,

        # Inner Numeric Constraints -- used to floating point under or overflow
        :floorToPreventOverflow => 1e-60 # 1e-30
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
        anExample = {:inputs => aPoint, :targets => (targets + aPoint), :exampleNumber => exampleNumber, :class => indexToClass}
        examples << anExample
      end
    end
    logger.puts "Error: Incorrect Number of Examples Generated and/or Specified" unless (examples.length == args[:numberOfExamples])
    examples
  end

  def createNetworkAndTrainer
    network = Flocking3LayerNetwork.new(args)
    theTrainer = TrainSuperCircleProblemBPFlockAndLocFlockAtOutputNeuron.new(examples, network, args)
    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

randomNumberSeed = 2

experiment = Experiment.new("J1SciBpFlock Science Net Architecture for Circle training", randomNumberSeed)

experiment.performSimulation()
