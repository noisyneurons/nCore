### VERSION "nCore"
## ../nCore/bin/NewCircleBPofFlockError.rb

require 'yaml'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralPartsExtended'
# require_relative '../lib/core/ExampleImportanceMods'    # TODO where should this go?
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers.rb'
require_relative '../lib/core/CorrectionForRateAtWhichNeuronsGainChanges'

def createTrainingSet(args)
  include ExampleDistribution

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
  STDERR.puts "****************Incorrect Number of Examples Specified!! ************************" if (args[:numberOfExamples] != examples.length)
  return examples
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
end

class Experiment
  def setParameters

    numberOfExamples = 24
    randomNumberSeed = 0

    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :rng => Random.new(randomNumberSeed),

        ## ??
        :phase1Epochs => 1e6, # 10000, # TODO check these 2 parameters for circle learning...
        :phase2Epochs => 0,
        ## ??

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.02,

        :minMSE => 0.001,
        :maxNumEpochs => 1e6,

        # Network Architecture and initial weights
        :numberOfInputNeurons => 2,
        :hiddenLayer1NumberOfNeurons => 10, # 20,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => numberOfExamples,
        :firstExamplesAngleToXAxis => 0.0,

        # Recording and database parameters
        :numberOfEpochsBetweenStoringDBRecords => 1000,

        # Flocking Parameters...
        :flockingLearningRate => -0.002,
        :maxFlockingIterationsCount => 2000, # 3800, # 2000,
        :maxAbsFlockingErrorsPerExample => 0.002, # 0.00000000000001, #0.002, # 0.005,   # 0.04 / numberOfExamples = 0.005

        :typeOfClusterer => DynamicClusterer,
        :numberOfClusters => 2,
        :m => 2.0,
        :numExamples => numberOfExamples,
        :exampleVectorLength => 1,
        :delta => 1e-2, # 1.0e-3,
        :maxNumberOfClusteringIterations => 10, # 100,
        :symmetricalCenters => true, # if true, speed is negatively affected

        # Inner Numeric Constraints -- used to floating point under or overflow
        :floorToPreventOverflow => 1e-30
    }
  end
end


###################################### Start of Main ##########################################
srand(0)
descriptionOfExperiment = "New Training Alg for CircleBPofFlockError"
experiment = ExperimentLogger.new(descriptionOfExperiment)
args = experiment.setParameters

############################### create training set...
examples = createTrainingSet(args)

############################### create testing set...
args[:firstExamplesAngleToXAxis] = 15.0
testingExamples = createTrainingSet(args)

######################## Specify data store and experiment description....
dataStoreManager = SimulationDataStoreManager.create

######################## Create Network....
network = CircleFlockingNeuronNetwork.new(dataStoreManager, args)
puts network

############################### train ...
trainingSequence = TrainingSequence.create(network, args)
theTrainer = NewCircleTrainer.new(trainingSequence, network, args)


lastEpoch, lastTrainingMSE, accumulatedAbsoluteFlockingErrors = theTrainer.simpleLearningWithFlocking(examples)

arrayOfNeuronsToPlot = nil
theTrainer.displayTrainingResults(arrayOfNeuronsToPlot)

lastTestingMSE = theTrainer.oneForwardPassEpoch(testingExamples)

puts "############ Include Example Numbers #############"

4000.times do |epochNumber|
  selectedData = DetailedNeuronData.lookup { |q| q[:experimentNumber_epochs_neuron].eq({experimentNumber: $globalExperimentNumber, epochs: epochNumber,
                                                                                        neuron: 2}) }
  puts "For epoch number=\t#{epochNumber}" unless (selectedData.empty?)

  selectedData.each { |itemKey| puts DetailedNeuronData.values(itemKey) } unless (selectedData.empty?)

end

puts "####################################"

displayAndPlotResults(args, accumulatedAbsoluteFlockingErrors, dataStoreManager, lastEpoch, lastTestingMSE,
                      lastTrainingMSE, network, theTrainer, trainingSequence)

SnapShotData.new(descriptionOfExperiment, network, Time.now, lastEpoch, lastTrainingMSE, lastTestingMSE)
$globalExperimentNumber
selectedData = SnapShotData.lookup { |q| q[:experimentNumber_epochs].eq({experimentNumber: $globalExperimentNumber, epochs: lastEpoch}) }

selectedData = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(5) }
unless (selectedData.empty?)
  puts
  puts "Number\tDescription\tLastEpoch\tTrainMSE\tTestMSE\tTime"
  selectedData.each do |aSelectedExperiment|
    aHash = SnapShotData.values(aSelectedExperiment)
    puts "#{aHash[:experimentNumber]}\t#{aHash[:descriptionOfExperiment]}\t#{aHash[:epochs]}\t#{aHash[:trainMSE]}\t#{aHash[:testMSE]}\t#{aHash[:time]}"
  end
end

DetailedNeuronData.deleteTables
experiment.save
