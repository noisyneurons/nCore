### VERSION "nCore"
## ../nCore/lib/core/SimulationDataStore.rb

require_relative 'Utilities'


module RecordingAndPlottingRoutines

  def storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, dPrimes)
    self.elapsedTime = Time.now - startTime
    dataStoreManager.storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, dPrimes, elapsedTime)
  end

  def generatePlotForEachNeuron(arrayOfNeuronsToPlot)
    arrayOfNeuronsToPlot.each do |theNeuron|
      xAry, yAry = getZeroXingExampleSet(theNeuron)
      plotDotsWhereOutputGtPt5(xAry, yAry, theNeuron, trainingSequence.epochs)
    end
  end

  def oneForwardPassEpoch(testingExamples)
    trainingExamples = args[:examples]
    distributeSetOfExamples(testingExamples)
    testMSE = measureNeuralResponsesForTesting
    distributeSetOfExamples(trainingExamples) # restore training examples
    return testMSE
  end

  def getZeroXingExampleSet(theNeuron)
    trainingExamples = args[:examples]
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
    self.examples = trainingExamples
    [x0Array, x1Array]
  end

  def measureNeuralResponsesForTesting
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      neuronsWithInputLinks.each { |aNeuron| aNeuron.recordResponsesForExample }
    end
    mse = network.calcNetworksMeanSquareError
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
    examples = [{:inputs => [x0, x1], :exampleNumber => 0}]
    distributeDataToAnArrayOfNeurons(inputLayer, examples)
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(0) }
    return theNeuron.output
  end
end


class Experiment
  attr_reader :descriptionOfExperiment, :args

  $redis.setnx("experimentNumber", 0)
  @@number = $redis.get("experimentNumber")

  def Experiment.number
    @@number
  end

  #def Experiment.deleteTable
  #  $redis.del("experimentNumber")
  #end

  def initialize(experimentDescription = nil)
    $redis.incr("experimentNumber")
    @descriptionOfExperiment = descriptionOfExperiment
  end

  def save
    $redis.save
  end
end


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

  def initialize(descriptionOfExperiment, network, time, epochs, trainMSE, testMSE = 0.0)
    @id = @@ID
    @@ID += 1
    @experimentNumber = Experiment.number

    @descriptionOfExperiment = descriptionOfExperiment
    @network = network.to_s
    @time = time
    @epochs = epochs
    @trainMSE = trainMSE
    @testMSE = testMSE

    theData = {:experimentNumber => experimentNumber, :descriptionOfExperiment => descriptionOfExperiment,
               :network => network.to_s,
               :time => time,
               :epochs => epochs,
               :trainMSE => trainMSE,
               :testMSE => testMSE
    }

    $redis.set(dataKey, YAML.dump(theData))
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
  end

  @@ID = 0

  def DetailedNeuronData.values(key)
    redisKey = key
    $redis.get(redisKey)
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
    @experimentNumber = Experiment.number
    @epochs = detailedNeuronDataToRecord[:epochs]
    @neuron = detailedNeuronDataToRecord[:neuronID]
    @exampleNumber = detailedNeuronDataToRecord[:exampleNumber]
    $redis.set(detailedNeuronDataKey, detailedNeuronDataToRecord)
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
  end

  @@ID = 0

  def NeuronData.values(key)
    $redis.get(key)
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
    @experimentNumber = Experiment.number
    @neuron = neuronDataToRecord[:neuronID]
    @epochs = neuronDataToRecord[:epochs]
    $redis.set(neuronDataKey, neuronDataToRecord)
    index!
  end

  def neuronDataKey
    "ND#{experimentNumber}.#{id}"
  end
end


module DBAccess
  def dbStoreData
    savingInterval = args[:intervalForSavingNeuronData]
    if recordOrNot?(savingInterval)
      aHash = metricRecorder.dataToRecord
      aHash.delete(:exampleNumber)
      aHash[:epochs] = args[:epochs]
      aHash[:accumulatedAbsoluteFlockingError] = accumulatedAbsoluteFlockingError
      NeuronData.new(aHash)
    end
  end

  def dbStoreDetailedData
    savingInterval = args[:intervalForSavingDetailedNeuronData]
    if recordOrNot?(savingInterval)
      aHash = metricRecorder.dataToRecord
      aHash[:epochs] = args[:epochs]
      DetailedNeuronData.new(aHash)
    end
  end

  def recordOrNot?(recordingInterval)
    epochs = args[:epochs]
    return ((epochs % recordingInterval) == 0)
  end

  #  def timeToRecordData
  #    epochs = args[epochs]
  #    record = false
  #    return if (epochs < 0)
  #    record = true if ((epochs % numberOfEpochsBetweenStoringDBRecords) == 0)
  #    record = true if lastEpoch
  #    return record
  #  end
end

class Neuron
  include DBAccess
end

class OutputNeuron
  include DBAccess
end


class SimulationDataStoreManager
  attr_accessor :args

  def initialize(args={})
    @args = args
  end

  def deleteDataForExperiment(experimentNumber)
    DetailedNeuronData.deleteData(experimentNumber)
    NeuronData.deleteData(experimentNumber)
  end

  def deleteAllDataAndIndexesExceptSnapShot!
    nextExperimentNumber = $redis.get("experimentNumber")
    (1...nextExperimentNumber.to_i).each do |experimentNumber|
      deleteDataForExperiment(experimentNumber)
    end
    DetailedNeuronData.deleteEntireIndex!
    NeuronData.deleteEntireIndex!
  end
end


