### VERSION "nCore"
## ../nCore/bin/CircleBPofFlockError.rb

require 'yaml'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralPartsExtended'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers.rb'

class SimpleFlockingNeuronNetwork
  def designateAndLabelGroupsOfNeurons
    @allNeuronsInOneArray = allNeuronLayers.flatten
    @inputLayer = allNeuronLayers[0]
    @hiddenLayer = allNeuronLayers[1]
    @outputLayer = allNeuronLayers[2]
    @neuronsWithInputLinks = hiddenLayer + outputLayer
    @neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
  end
end

def generateExamplesOnConcentricCircles(firstExamplesAngleToXAxis, args)
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

  dataSetFromJoin = dataStoreManager.joinDataSets # joinForShoesDisplay
  dataStoreManager.transferDataSetToVisualizer(dataSetFromJoin, args)
end

def setParameters(descriptionOfExperiment)
  numberOfExamples = 24
  args = {
      :descriptionOfExperiment => descriptionOfExperiment,

      # parameters that impact learning dynamics:
      :learningRate => 'Not Used!',

      :learningRateNoFlockPhase1 => 0.03,
      :learningRateLocalFlockPhase2 => -0.002,
      :learningRateForBackPropedFlockingErrorPhase2 => -0.002,

      :phase1Epochs => 100, #100, # 10,
      :phase2Epochs => 200, # 10,

      # Stop training parameters

      :minMSE => 0.001, # 0.01,
      :maxNumEpochs => 1e6, # 1e5,

      # Network Architecture and initial weights

      :numberOfInputNeurons => 2,
      :numberOfHiddenNeurons => 10, # 20,
      :numberOfOutputNeurons => 1,
      :weightRange => 1.0,
      :typeOfLink => FlockingLink,

      # Training Set parameters
      :numberOfExamples => numberOfExamples,
      :rightShiftUpper2Classes => 0.0,

      # Recording and database parameters
      :numberOfEpochsBetweenStoringDBRecords => 1000,

      # Flocking Parameters...
      :typeOfClusterer => DynamicClusterer,
      :numberOfClusters => 2,
      :m => 2.0,
      :numExamples => numberOfExamples,
      :exampleVectorLength => 2,
      :delta => 1.0e-3,
      :maxNumberOfClusteringIterations => 100,
      :symmetricalCenters => false, # if true, speed is negatively affected

      # Inner Numeric Constraints
      :minDistanceAllowed => 1.0e-30,
      :leadingFactor => 1.0
  }
end


###################################### Start of Main ##########################################
srand(0)
descriptionOfExperiment = "CircleBPofFlockError Reference Run"
args = setParameters(descriptionOfExperiment)

############################### create training set...
examples = generateExamplesOnConcentricCircles(firstExamplesAngleToXAxis = 0.0, args)
testingExamples = generateExamplesOnConcentricCircles(firstExamplesAngleToXAxis = 15.0, args)

######################## Specify data store and experiment description....
databaseFilename = "acrossEpochsSequel" #  = ""
dataStoreManager = SimulationDataStoreManager.create(databaseFilename, examples, args)

######################## Create Network and Train....
network = SimpleFlockingNeuronNetwork.new(dataStoreManager, args)

trainingSequence = TrainingSequence.create(network, args)
theTrainer = CircleTrainer.new(trainingSequence, network, args)
lastEpoch, lastTrainingMSE, dPrimes = theTrainer.simpleLearningWithFlocking(examples)

lastTestingMSE = theTrainer.oneForwardPassEpoch(testingExamples)
theTrainer.storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, dPrimes)

displayAndPlotResults(args, dPrimes, dataStoreManager, lastEpoch, lastTestingMSE,
                      lastTrainingMSE, network, theTrainer, trainingSequence)


