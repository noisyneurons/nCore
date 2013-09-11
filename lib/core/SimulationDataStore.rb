### VERSION "nCore"
## ../nCore/lib/core/SimulationDataStore.rb

require_relative 'Utilities'


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

  def initialize(experimentDescription)
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


class FlockData
  include Relix
  Relix.host = $currentHost
  attr_accessor :id, :experimentNumber, :epochs, :neuron, :exampleNumber, :netInputs, :flockErrors

  relix do
    primary_key :flockDataKey
    multi :experimentNumber_epochs_neuron, on: %w(experimentNumber epochs neuron)
    multi :experimentNumber, index_values: true
  end

  @@ID = 0

  def FlockData.values(key)
    redisKey = key
    $redis.get(redisKey)
  end

  def FlockData.deleteData(experimentNumber)
    ary = $redis.keys("FD#{experimentNumber}*")
    ary.each { |item| $redis.del(item) }
  end

  def FlockData.deleteEntireIndex!
    ary = $redis.keys("FlockData*")
    ary.each { |item| $redis.del(item) }
  end

  def initialize(epochs, neuron, exampleNumber, netInputs=1, flockErrors=1)
    @id = @@ID
    @@ID += 1
    @experimentNumber = Experiment.number
    @epochs = epochs
    @neuron = neuron
    @netInputs = netInputs
    @flockErrors = flockErrors
    @exampleNumber = exampleNumber

    theData = {:experimentNumber => experimentNumber, :epochs => epochs, :neuron => neuron,
               :exampleNumber => exampleNumber, :netInputs => netInputs, :flockErrors => flockErrors}

    $redis.set(flockDataKey, theData)

    index!
  end

  def flockDataKey
    "FD#{experimentNumber}.#{id}"
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
    @epochs = neuronDataToRecord[:epochNumber]
    $redis.set(neuronDataKey, neuronDataToRecord)
    index!
  end

  def neuronDataKey
    "ND#{experimentNumber}.#{id}"
  end
end


class SimulationDataStoreManager
  attr_accessor :args, :epochNumber

  def SimulationDataStoreManager.instance
    return @@dataStoreManager
  end

  def initialize(args)
    @args = args
    @epochNumber = nil
  end

  def deleteDataForExperiment(experimentNumber)
    FlockData.deleteData(experimentNumber)
    NeuronData.deleteData(experimentNumber)
  end
end


