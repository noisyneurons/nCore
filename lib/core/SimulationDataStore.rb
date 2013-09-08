### VERSION "nCore"
## ../nCore/lib/core/SimulationDataStore.rb


require_relative 'Utilities'

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

  def SnapShotData.deleteTables
    ary = $redis.keys("SSD*")
    ary.each { |item| $redis.del(item) }
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

  def FlockData.deleteTables
    ary = $redis.keys("FD*")
    ary.each { |item| $redis.del(item) }
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

  def NeuronData.deleteTables
    ary = $redis.keys("ND*")
    ary.each { |item| $redis.del(item) }
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
  attr_accessor :epochNumber
  private_class_method(:new)

  @@dataStoreManager = nil

  def SimulationDataStoreManager.create
    @@dataStoreManager = new unless @@dataStoreManager
    @@dataStoreManager
  end

  def SimulationDataStoreManager.instance
    return @@dataStoreManager
  end

  def initialize
    STDERR.puts "ERROR:  SimulationDataStoreManager.new called MORE THAN ONE TIME!!!!!" unless (@@dataStoreManager.nil?)
    @epochNumber = nil
    @@dataStoreManager = self
  end

  def deleteTemporaryTables
    FlockData.deleteTables
    NeuronData.deleteTables
  end
end


