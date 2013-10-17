### VERSION "nCore"
## ../nCore/bin/CircleBPofFlockError.rb

require_relative 'BaseLearningExperiment'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

class Experiment

  def setParameters

    args = {
        :experimentNumber => ExperimentLogger.number,
        :descriptionOfExperiment => descriptionOfExperiment,
        :rng => Random.new(randomNumberSeed),

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.3, # 0.03,
        :minMSE => 0.001,
        :maxNumEpochs => 8e3,


        # Network Architecture and initial weights
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 20, # 20,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => (numberOfExamples = 24),
        :firstExamplesAngleToXAxis => 0.0,
        :firstExamplesAngleToXAxisForTesting => 15.0,

        # Recording and database parameters
        :intervalForSavingNeuronData => 1000,
        :intervalForSavingDetailedNeuronData => 5000,
        :intervalForSavingTrainingData => 1000,

        # Flocking Parameters...
        :flockingLearningRate => -0.001, # -0.0002,
        # :learningRateForBackPropedFlockingError => -0.002,
        :maxFlockingIterationsCount => 100, # 2000,
        :targetFlockIterationsCount => 20,
        :ratioDropInMSE => 0.95,
        :ratioDropInMSEForFlocking => 0.96,
        # :maxAbsFlockingErrorsPerExample => 0.2, #  0.04 / numberOfExamples = 0.005

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
        anExample = {:inputs => aPoint, :targets => targets, :exampleNumber => exampleNumber, :class => indexToClass}
        examples << anExample
      end
    end
    STDERR.puts "Error: Incorrect Number of Examples Generated and/or Specified" unless (examples.length == args[:numberOfExamples])
    examples
  end

  def createNetworkAndTrainer
    network = Flocking3LayerNetwork.new(args)
    theTrainer = TrainingSupervisor3LayersOutputNeuronLocalFlocking.new(examples, network, args)
    return network, theTrainer
  end
end


###################################### START of Main Learning  ##########################################

experiment = Experiment.new("CircleBPofFlockError using correctionFactorForRateAtWhichNeuronsGainChanges", randomNumberSeed=0)

experiment.performSimulation()
