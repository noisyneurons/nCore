### VERSION "nCore"
## ../nCore/bin/TunedAnalogy4Class.rb

require 'yaml'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralPartsExtended'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers.rb'


### Mods for reporting
class FlockingNeuronRecorder ##  TODO temporary
  def recordResponsesForEpoch
    if (trainingSequence.timeToRecordData)
      determineCentersOfClusters()
      epochDataToRecord = ({:epochNumber => dataStoreManager.epochNumber, :neuronID => neuron.id,
                            :wt1 => neuron.inputLinks[0].weight, :wt2 => neuron.inputLinks[1].weight,
                            :cluster0Center => @cluster0Center, :cluster1Center => @cluster1Center,
                            :dPrime => neuron.dPrime})
      #quickReportOfExampleWeightings(epochDataToRecord)
      epochDataSet.insert(epochDataToRecord)
    end
  end

  def quickReportOfExampleWeightings(epochDataToRecord)
    neuron.clusters.each_with_index do |cluster, numberOfCluster|
      cluster.membershipWeightForEachExample.each { |exampleWeight| puts "Epoch Number, Cluster Number and Example Weighting= #{epochDataToRecord[:epochNumber]}\t#{numberOfCluster}\t#{exampleWeight}" }
      puts
    end
  end
end
# Mods for reporting

def createTrainingSet(args)
  numberOfExamples = args[:numberOfExamples]
  rightShiftUpper2Classes = args[:rightShiftUpper2Classes]

  xStart = [-1.0+rightShiftUpper2Classes, 1.0+rightShiftUpper2Classes, 1.0, -1.0] # assumes clockwise numbering of classes, from 10:30 being class 0
  yStart = [1.0, 1.0, -1.0, -1.0]


  xInc = [0.0, 0.0, 0.0, 0.0]
  yInc = [1.0, 1.0, -1.0, -1.0]

  # target = [0.9, 0.9, 0.9, 0.9] # potentially will use this example set for supervised learning.

  numberOfClasses = xStart.length
  numberOfExamplesInEachClass = numberOfExamples / numberOfClasses
  exampleNumber = 0

  examples = []
  numberOfClasses.times do |classOfExample|
    xS = xStart[classOfExample]
    xI = xInc[classOfExample]
    yS = yStart[classOfExample]
    yI = yInc[classOfExample]
    numberOfExamplesInEachClass.times do |classExNumb|
      x = xS + (xI * classExNumb)
      y = yS + (yI * classExNumb)
      aPoint = [x, y]
      targets = [0.0, 0.0, 0.0, 0.0]
      targets[classOfExample] = 1.0
      examples << {:inputs => aPoint, :targets => targets, :exampleNumber => exampleNumber, :class => classOfExample}
      exampleNumber += 1
    end
  end
  STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  examples
end

def displayAndPlotResults(args, dPrimes, dataStoreManager, lastEpoch,
    lastTestingMSE, lastTrainingMSE, network, theTrainer, trainingSequence)

  puts network
  puts
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
  numberOfExamples = 16
  randomNumberSeed = 0
  args = {
      :descriptionOfExperiment => descriptionOfExperiment,

      :rng => Random.new(randomNumberSeed),

      :maxHistory => 8,
      :balanceOfdPrimeVsDispersion => 0.0, # a value of 1.0 indicates that dPrime is
      # to be the sole metric. a value of 0.0 indicates Dispersion is the sole metric
      :multiplyToEmphasizeFlocking => 0.5, # if value = 0.0 only output error
      # is used to determine weight changes.  If value >> 1.0, then flocking error will
      # be dominant in prescribing weight changes.
      :searchRangeRatio => 2.0,

      # parameters that impact learning dynamics:
      #:learningRate => 1.0,
      #:learningRateNoFlockPhase1 => 3.0,
      #:learningRateLocalFlockPhase2 => -0.005,
      #:learningRateForBackPropedFlockingErrorPhase2 => -0.005,
      #:learningRateBPOutputErrorPhase2 => 0.5,

      :phase1Epochs => 80,
      :phase2Epochs => 0,

      # Stop training parameters
      :minMSE => 0.0001,
      :maxNumEpochs => 80, # 4.0e1,

      # Network Architecture and initial weights
      :numberOfInputNeurons => 2,
      :layer1NumberOfHiddenNeurons => 2,
      :layer2NumberOfHiddenNeurons => 2,
      :layer3NumberOfHiddenNeurons => 2,
      :numberOfOutputNeurons => 4,
      :weightRange => 1.0,
      :typeOfLink => FlockingLink,

      # Training Set parameters
      :numberOfExamples => numberOfExamples,
      :rightShiftUpper2Classes => 0.5, # 0.5,

      # Recording and database parameters
      :numberOfEpochsBetweenStoringDBRecords => 500,

      # Flocking Parameters...
      :typeOfClusterer => DynamicClusterer,
      :numberOfClusters => 2,
      :m => 2.0, #1.1, # 2.0
      :numExamples => numberOfExamples,
      :exampleVectorLength => 2,
      :delta => 1e-3,
      :maxNumberOfClusteringIterations => 100,
      :symmetricalCenters => false, # if true, speed is negatively affected

      # Inner Numeric Constraints
      :leadingFactor => 1.0, # 1.02,

      # Inner Numeric Constraints
      :floorToPreventOverflow => 1e-30
  }
end

###################################### Start of Main ##########################################
srand(0)
descriptionOfExperiment = "TunedAnalogy4Class NoBPofFlockingError multi-step, 2-phase"
args = setParameters(descriptionOfExperiment)

############################### create training set...
examples = createTrainingSet(args)

######################## Specify data store and experiment description....
databaseFilename = "acrossEpochsSequel" #  = ""
dataStoreManager = SimulationDataStoreManager.create(databaseFilename, examples, args)

######################## Create Network and Train....
network = AnalogyNetwork.new(dataStoreManager, args)
# network = AnalogyNetworkNoJumpLinks.new(dataStoreManager, args)

trainingSequence = TrainingSequence.create(network, args)

theTrainer = TunedTrainerAnalogy4ClassNoBPofFlockError.new(trainingSequence, network, args)

arrayOfNeuronsForIOPlots = [network.hiddenLayer1[0], network.hiddenLayer1[1], network.hiddenLayer2[0], network.hiddenLayer2[1], network.outputLayer[0], network.outputLayer[1]]
# arrayOfNeuronsForIOPlots = []
lastEpoch, lastTrainingMSE, dPrimes = theTrainer.simpleLearningWithFlocking(examples, arrayOfNeuronsForIOPlots)

theTrainer.storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE=nil, dPrimes)

displayAndPlotResults(args, dPrimes, dataStoreManager, lastEpoch, lastTestingMSE,
                      lastTrainingMSE, network, theTrainer, trainingSequence)


