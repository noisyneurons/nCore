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
        :outputErrorLearningRate => 0.03,
        :minMSE => 0.001,
        :maxNumEpochs => 1e6,


        # Network Architecture and initial weights
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 10, # 20,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => (numberOfExamples = 24),
        :firstExamplesAngleToXAxis => 0.0,

        # Recording and database parameters
        :intervalForSavingNeuronData => 100,
        :intervalForSavingDetailedNeuronData => 1000,
        :intervalForSavingTrainingData => 100,

        # Flocking Parameters...
        :flockingLearningRate => -0.002, # -0.002,
        :learningRateForBackPropedFlockingError => -0.002,
        :maxFlockingIterationsCount => 10, # 2000,
        :maxAbsFlockingErrorsPerExample => 0.002, #  0.04 / numberOfExamples = 0.005

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
        :epochsBeforeFlockingAllowed => 200,

        # Inner Numeric Constraints -- used to floating point under or overflow
        :floorToPreventOverflow => 1e-60 # 1e-30
    }
  end

  def createTrainingSet
    firstExamplesAngleToXAxis = args[:firstExamplesAngleToXAxis]
    numberOfExamples = args[:numberOfExamples]

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
    STDERR.puts "Error: Incorrect Number of Examples Generated and/or Specified" unless(examples.length == args[:numberOfExamples])
    return examples
  end
end


def displayAndPlotResults(args, dPrimes, dataStoreManager, lastEpoch,
    lastTestingMSE, lastTrainingMSE, network, theTrainer, trainingSequence)
  puts network
  puts "Elapsed Time=\t#{theTrainer.elapsedTime}"
  puts "\tAt Epoch #{trainingSequence.epochs}"
  puts "\tAt Epoch #{lastEpoch}"
  puts "\t\tThe Network's Training MSE=\t#{lastTrainingMSE}\t and TEST MSE=\t#{lastTestingMSE}\n"
  puts "\t\t\tThe dPrime(s) at the end of training are: #{dPrimes}"

#############################  plotting and visualization....
  plotMSEvsEpochNumber(network)

  dataSetFromJoin = dataStoreManager.joinDataSets # joinForShoesDisplay
  dataStoreManager.transferDataSetToVisualizer(dataSetFromJoin, args)
end



###################################### START of Main Learning  ##########################################

experiment = Experiment.new("CircleBPofFlockError using correctionFactorForRateAtWhichNeuronsGainChanges", randomNumberSeed=0)

experiment.performSimulation()
