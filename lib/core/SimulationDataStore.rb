### VERSION "nCore"
## ../nCore/lib/core/SimulationDataStore.rb

require_relative 'Utilities'

module RecordingAndPlottingRoutines

  #def measureNeuralResponsesForTesting
  #  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  #  numberOfExamples.times do |exampleNumber|
  #    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
  #    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
  #    neuronsWithInputLinks.each { |aNeuron| aNeuron.recordResponsesForExample }
  #  end
  #  mse = network.calcNetworksMeanSquareError
  #end

  #def oneForwardPassEpoch(testingExamples)
  #  trainingExamples = args[:examples]
  #  distributeSetOfExamples(testingExamples)
  #  testMSE = measureNeuralResponsesForTesting
  #  distributeSetOfExamples(trainingExamples) # restore training examples
  #  return testMSE
  #end

  def generatePlotForEachNeuron(arrayOfNeuronsToPlot)
    arrayOfNeuronsToPlot.each do |theNeuron|
      xAry, yAry = getZeroXingExampleSet(theNeuron)
      plotDotsWhereOutputGtPt5(xAry, yAry, theNeuron, args[:epochs])
    end
  end

  def getZeroXingExampleSet(theNeuron)
    minX1 = -4.0
    maxX1 = 4.0
    numberOfTrials = 100
    increment = (maxX1 - minX1) / numberOfTrials
    x0Array = []
    x1Array = []
    numberOfTrials.times do |index|
      x1 = minX1 + index * increment
      x0 = findZeroCrossingFast(x1, theNeuron)
      next if (x0.nil?)
      x1Array << x1
      x0Array << x0
    end
    [x0Array, x1Array]
  end

  def findZeroCrossingFast(x1, theNeuron)
    minX0 = -2.0
    maxX0 = 2.0
    range = maxX0 - minX0

    initialOutput = ioFunctionFor2inputs(minX0, x1, theNeuron)
    initialOutputLarge = false
    initialOutputLarge = true if (initialOutput >= 0.5)

    wasThereACrossing = false
    20.times do |count|
      range = maxX0 - minX0
      bisectedX0 = minX0 + (range / 2.0)
      currentOutput = ioFunctionFor2inputs(bisectedX0, x1, theNeuron)
      currentOutputLarge = false
      currentOutputLarge = true if (currentOutput > 0.5)
      if (currentOutputLarge != initialOutputLarge)
        maxX0 = bisectedX0
        wasThereACrossing = true
      else
        minX0 = bisectedX0
      end
    end
    return nil if (wasThereACrossing == false)
    estimatedCrossingForX0 = (minX0 + maxX0) / 2.0
    return estimatedCrossingForX0
  end

  def ioFunctionFor2inputs(x0, x1, theNeuron) # TODO should this be moved to an analysis or plotting class?
    clampOutputsOfTheInputNeuronsThatDrive(x0, x1, theNeuron)
    theNeuron.propagate(0)
    return theNeuron.output
  end

  def clampOutputsOfTheInputNeuronsThatDrive(x0, x1, theNeuron)
    inputLinks = theNeuron.inputLinks
    neuronsDrivingTheNeuron = []
    inputLinks.length.times do |index|
      neuronsDrivingTheNeuron << inputLinks[index].inputNeuron
    end
    index = 0
    neuronsDrivingTheNeuron[index].output = x0
    index += 1
    neuronsDrivingTheNeuron[index].output = x1
  end
end

#############################

class ExperimentLogger
  attr_reader :descriptionOfExperiment, :experimentNumber, :args

  $redis.setnx("experimentNumber", 0)

  def deleteTemporaryDataRecordsInDB
    TrainingData.deleteData(experimentNumber)
    TrainingData.deleteEntireIndex!
    NeuronData.deleteData(experimentNumber)
    NeuronData.deleteEntireIndex!
    DetailedNeuronData.deleteData(experimentNumber)
    DetailedNeuronData.deleteEntireIndex!
  end

  def ExperimentLogger.deleteTable!
    $redis.del("experimentNumber")
  end

  def initialize(experimentDescription = nil)
    @experimentNumber = $redis.incr("experimentNumber")
    @descriptionOfExperiment = descriptionOfExperiment
  end

  def save
    $redis.save
  end
end

#############################

class SnapShotData
  include Relix
  Relix.host = $currentHost
  attr_accessor :id, :experimentNumber, :descriptionOfExperiment, :network, :time, :epochs, :trainMSE, :testMSE

  relix do
    primary_key :dataKey
    ordered :experimentNumber
    multi :experimentNumber_epochs, on: %w(experimentNumber epochs)
    # multi :experimentNumber, index_values: true
  end

  @@ID = 0

  def SnapShotData.values(key)
    YAML.load($redis.get(key))
  end

  def SnapShotData.deleteData(experimentNumber)
    ary = $redis.keys("SSD#{experimentNumber}*")
    ary.each { |item| $redis.del(item) }
  end

  def SnapShotData.deleteEntireIndex!
    ary = $redis.keys("SnapShotData*")
    ary.each { |item| $redis.del(item) }
  end

  def initialize(dataToRecord)
    @id = @@ID
    @@ID += 1
    @experimentNumber = dataToRecord[:experimentNumber]
    @epochs = dataToRecord[:epochs]

    $redis.set(dataKey, YAML.dump(dataToRecord))
    index!
  end

  def dataKey
    "SSD#{experimentNumber}.#{id}"
  end
end

class DetailedNeuronData
  include Relix
  Relix.host = $currentHost
  attr_accessor :id, :experimentNumber, :epochs, :neuron, :exampleNumber

  relix do
    primary_key :detailedNeuronDataKey
    multi :experimentNumber_epochs_neuron_exampleNumber, on: %w(experimentNumber epochs neuron exampleNumber)
    multi :experimentNumber, index_values: true
    multi :epochs, index_values: true
    multi :exampleNumber, index_values: true
  end

  @@ID = 0

  def DetailedNeuronData.values(key)
    YAML.load($redis.get(key))
  end

  def DetailedNeuronData.deleteData(experimentNumber)
    ary = $redis.keys("DND#{experimentNumber}*")
    ary.each { |item| $redis.del(item) }
  end

  def DetailedNeuronData.deleteEntireIndex!
    ary = $redis.keys("DetailedNeuronData*")
    ary.each { |item| $redis.del(item) }
  end

  def initialize(detailedNeuronDataToRecord)
    @id = @@ID
    @@ID += 1
    @experimentNumber = $globalExperimentNumber
    @epochs = detailedNeuronDataToRecord[:epochs]
    @neuron = detailedNeuronDataToRecord[:neuronID]
    @exampleNumber = detailedNeuronDataToRecord[:exampleNumber]
    $redis.set(detailedNeuronDataKey, YAML.dump(detailedNeuronDataToRecord))
    index!
  end

  def detailedNeuronDataKey
    "DND#{experimentNumber}.#{id}"
  end
end

class NeuronData
  include Relix
  Relix.host = $currentHost
  attr_accessor :id, :experimentNumber, :epochs, :neuron

  relix do
    primary_key :neuronDataKey
    multi :experimentNumber_epochs_neuron, on: %w(experimentNumber epochs neuron)
    multi :experimentNumber, index_values: true
    multi :epochs, index_values: true
  end

  @@ID = 0

  def NeuronData.values(key)
    YAML.load($redis.get(key))
  end

  def NeuronData.deleteData(experimentNumber)
    ary = $redis.keys("ND#{experimentNumber}*")
    ary.each { |item| $redis.del(item) }
  end

  def NeuronData.deleteEntireIndex!
    ary = $redis.keys("NeuronData*")
    ary.each { |item| $redis.del(item) }
  end

  def initialize(neuronDataToRecord)
    @id = @@ID
    @@ID += 1
    @experimentNumber = $globalExperimentNumber
    @neuron = neuronDataToRecord[:neuronID]
    @epochs = neuronDataToRecord[:epochs]
    $redis.set(neuronDataKey, YAML.dump(neuronDataToRecord))
    index!
  end

  def neuronDataKey
    "ND#{experimentNumber}.#{id}"
  end
end

class TrainingData
  include Relix
  Relix.host = $currentHost
  attr_accessor :id, :experimentNumber, :epochs

  relix do
    primary_key :trainingDataKey
    multi :experimentNumber_epochs, on: %w(experimentNumber epochs)
    multi :experimentNumber, index_values: true
    multi :epochs, index_values: true
  end

  @@ID = 0

  def TrainingData.values(key)
    YAML.load($redis.get(key))
  end

  def TrainingData.deleteData(experimentNumber)
    ary = $redis.keys("TD#{experimentNumber}*")
    ary.each { |item| $redis.del(item) }
  end

  def TrainingData.deleteEntireIndex!
    ary = $redis.keys("TrainingData*")
    ary.each { |item| $redis.del(item) }
  end

  def initialize(trainingDataToRecord)
    @id = @@ID
    @@ID += 1
    @experimentNumber = $globalExperimentNumber
    @epochs = trainingDataToRecord[:epochs]
    $redis.set(trainingDataKey, YAML.dump(trainingDataToRecord))
    index!
  end

  def trainingDataKey
    "TD#{experimentNumber}.#{id}"
  end
end

#####################################

module DBAccess
  def dbStoreNeuronData
    savingInterval = args[:intervalForSavingNeuronData]
    if recordOrNot?(savingInterval)
      aHash = metricRecorder.dataToRecord
      aHash.delete(:exampleNumber)
      aHash.delete(:error)
      aHash[:epochs] = args[:epochs]
      aHash[:accumulatedAbsoluteFlockingError] = accumulatedAbsoluteFlockingError
      NeuronData.new(aHash)
    end
  end

  def dbStoreDetailedData
    savingInterval = args[:intervalForSavingDetailedNeuronData]
    if recordOrNot?(savingInterval)
      aHash = metricRecorder.dataToRecord
      aHash.delete(:error)
      aHash[:epochs] = args[:epochs]
      DetailedNeuronData.new(aHash)
    end
  end

  def dbStoreTrainingData
    savingInterval = args[:intervalForSavingTrainingData]
    if recordOrNot?(savingInterval)
      trainMSE = calcMeanSumSquaredErrors
      testMSE = calcTestingMeanSquaredErrors
      puts "epoch number = #{args[:epochs]}\ttrainMSE = #{trainMSE}\ttestMSE = #{testMSE}"
      aHash = {:experimentNumber => $globalExperimentNumber, :epochs => args[:epochs], :mse => trainMSE, :testMSE => testMSE, :accumulatedAbsoluteFlockingErrors => accumulatedAbsoluteFlockingErrors}
      TrainingData.new(aHash)
    end
  end

  def recordOrNot?(recordingInterval)
    epochs = args[:epochs]
    return true if (epochs == 0)
    return (((epochs + 1) % recordingInterval) == 0)
  end
end

class Neuron
  include DBAccess
end

class OutputNeuron
  include DBAccess
end

class AbstractStepTrainer
  include DBAccess
end

#####################################

class SimulationDataStoreManager
  attr_accessor :args

  def initialize(args={})
    @args = args
  end

  def deleteDataForExperiment(experimentNumber)
    DetailedNeuronData.deleteData(experimentNumber)
    NeuronData.deleteData(experimentNumber)
    TrainingData.deleteData(experimentNumber)
  end

  def deleteAllDataAndIndexesExceptSnapShot!
    nextExperimentNumber = $redis.get("experimentNumber")
    (1...nextExperimentNumber.to_i).each do |experimentNumber|
      deleteDataForExperiment(experimentNumber)
    end
    DetailedNeuronData.deleteEntireIndex!
    NeuronData.deleteEntireIndex!
    TrainingData.deleteEntireIndex!
  end
end

#def recordResponsesForEpoch
#  if (trainingSequence.timeToRecordData)
#    determineCentersOfClusters()
#    epochDataToRecord = ({:epochNumber => dataStoreManager.epochNumber, :neuronID => neuron.id,
#                          :wt1 => neuron.inputLinks[0].weight, :wt2 => neuron.inputLinks[1].weight,
#                          :cluster0Center => @cluster0Center, :cluster1Center => @cluster1Center,
#                          :dPrime => neuron.dPrime})
#    quickReportOfExampleWeightings(epochDataToRecord)
#    NeuronData.new(epochDataToRecord)
#  end
#end
#
#def quickReportOfExampleWeightings(epochDataToRecord)
#  neuron.clusters.each_with_index do |cluster, numberOfCluster|
#    cluster.membershipWeightForEachExample.each { |exampleWeight| puts "Epoch Number, Cluster Number and Example Weighting= #{epochDataToRecord[:epochNumber]}\t#{numberOfCluster}\t#{exampleWeight}" }
#    puts
#    puts "NumExamples=\t#{cluster.numExamples}\tNum Membership Weights=\t#{cluster.membershipWeightForEachExample.length}"
#  end
#end
#
#def determineCentersOfClusters
#  cluster0 = neuron.clusters[0]
#  if (cluster0.center.present?)
#    @cluster0Center = cluster0.center[0]
#    cluster1 = neuron.clusters[1]
#    @cluster1Center = cluster1.center[0]
#  else
#    cluster0Center = 0.0
#    cluster1Center = 0.0
#  end
#end
