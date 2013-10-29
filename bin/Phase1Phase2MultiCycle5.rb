### VERSION "nCore"
## ../nCore/bin/Phase1Phase2MultiCycle5.rb

require 'yaml'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralPartsExtended'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'
require_relative '../lib/core/Trainers.rb'


#module CommonNeuronCalculations
#  public
#  def clearWithinEpochMeasures
#    metricRecorder.clearWithinEpochMeasures
#    upperErAry.clear
#    flockErAry.clear
#  end
#end
#
#class FlockingOutputNeuron
#  attr_accessor :upperErAry, :flockErAry, :flockErrorMultiplier
#
#  def postInitialize
#    self.output = ioFunction(@netInput = 0.0) # Only doing this in case we wish to use this code for recurrent networks
#    @inputLinks = []
#    @netInput = 0.0
#    @higherLayerError = 0.0
#    @errorToBackPropToLowerLayer = 0.0
#    @localFlockingError = 0.0
#    @arrayOfSelectedData = nil
#    @keyToExampleData = :targets
#    @exampleNumber = nil
#    @metricRecorder= FlockingOutputNeuronRecorder.new(self, args)
#    typeOfClusterer = args[:typeOfClusterer]
#    @clusterer = typeOfClusterer.new(args)
#    @store = {}
#    @dPrime = 0.0
#    @trainingSequence = TrainingSequence.instance
#    @upperErAry = []
#    @flockErAry = []
#    @flockErrorMultiplier = 0.0
#  end
#end
#
#class FlockingNeuron
#  attr_accessor :upperErAry, :flockErAry, :flockErrorMultiplier
#
#  def postInitialize
#    @inputLinks = []
#    @netInput = 0.0
#    self.output = self.ioFunction(@netInput) # Only doing this in case we wish to use this code for recurrent networks
#    @outputLinks = []
#    @higherLayerError = 0.0
#    @errorToBackPropToLowerLayer = 0.0
#    @localFlockingError = 0.0
#    @metricRecorder= FlockingNeuronRecorder.new(self, args)
#    @exampleNumber = nil
#    typeOfClusterer = args[:typeOfClusterer]
#    @clusterer = typeOfClusterer.new(args)
#    @store = {}
#    @dPrime = 0.0
#    @trainingSequence = TrainingSequence.instance
#    @upperErAry = []
#    @flockErAry = []
#    @flockErrorMultiplier = 0.0
#  end
#end

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

#class Trainer4Class
#  def designateNeuronGroups
#    neuronsCreatingFlockingError = hiddenLayer
#    adaptingNeurons = hiddenLayer
#    neuronsAdaptingToBackPropedFlockingError = []
#    return neuronsCreatingFlockingError, adaptingNeurons, neuronsAdaptingToBackPropedFlockingError
#  end
#end

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
  args = {
      :descriptionOfExperiment => descriptionOfExperiment,

      # parameters that impact learning dynamics:
      :learningRate => 'Not Used!', #1.0,

      :learningRateNoFlockPhase1 => 3.0,
      :learningRateLocalFlockPhase2 => -0.005,
      :learningRateBPOutputErrorPhase2 => 0.5,

      :phase1Epochs => 10,
      :phase2Epochs => 10,

      # Stop training parameters
      :minMSE => 0.001,
      :maxNumEpochs => 1.0e3,

      # Network Architecture and initial weights
      :numberOfInputNeurons => 2,
      :numberOfHiddenNeurons => 2,
      :numberOfOutputNeurons => 4,
      :weightRange => 1.0,

      # Training Set parameters
      :numberOfExamples => numberOfExamples,
      :rightShiftUpper2Classes => 0.0,

      # Recording and database parameters
      :numberOfEpochsBetweenStoringDBRecords => 10,

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
      :floorToPreventOverflow => 1.0e-30,
      :leadingFactor => 1.0 # 1.02,
  }
end

###################################### Start of Main ##########################################
srand(0)
descriptionOfExperiment = "Phase1Phase2MultiCycle5 Reference Run"
args = setParameters(descriptionOfExperiment)

############################### create training set...
examples = createTrainingSet(args)

######################## Specify data store and experiment description....
databaseFilename = "acrossEpochsSequel" #  = ""
dataStoreManager = SimulationDataStoreManager.create(databaseFilename, examples, args)

######################## Create Network and Train....
network = SimpleFlockingNeuronNetwork.new(dataStoreManager, args)

trainingSequence = TrainingSequence.create(network, args)
theTrainer = Trainer4Class.new(trainingSequence, network, args)
lastEpoch, lastTrainingMSE, dPrimes = theTrainer.simpleLearningWithFlocking(examples)

theTrainer.storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE=nil, dPrimes)

displayAndPlotResults(args, dPrimes, dataStoreManager, lastEpoch, lastTestingMSE,
                      lastTrainingMSE, network, theTrainer, trainingSequence)


