### VERSION "nCore"
## ../nCore/lib/core/SimulationDataStore.rb

require 'rubygems'

require_relative 'Utilities'
require 'relix'
require 'yaml'

$redis = Redis.new
# $redis.flushdb
$redis.setnx("experimentNumber",0)

class Experiment
  attr_reader :experimentNumber, :descriptionOfExperiment, :args

  def initialize(experimentDescription)
    @experimentNumber = $redis.get("experimentNumber").to_i
    $redis.incr("experimentNumber")
    @descriptionOfExperiment = descriptionOfExperiment
  end

  def setParameters
    STDERR.puts "Abstract method called"
  end
end


#puts "Experiment.new.experimentNumber=\t#{Experiment.new("hi").experimentNumber}"



class FlockData
  include Relix
  attr_accessor :experimentNumber, :id, :epochs, :neuron, :exampleNumber, :netInput, :flockError

  relix do
    primary_key :flockDataKey
    #ordered :epochs
    multi :epochs, order: :neuron
    multi :netInputs, order: :flockErrors
    # multi :neuron, order: :epochs

    multi :epochs, index_values: true
    multi :neuron, index_values: true
    multi :netInputs, index_values: true
    multi :flockErrors, index_values: true
  end


  @@ID = 0

  def initialize(experimentNumber, epochs, neuron, netInputs=1, flockErrors=1)
    @id = @@ID
    @@ID += 1
    @experimentNumber = experimentNumber
    @epochs = epochs
    @neuron = neuron
    @netInputs = netInputs
    @flockErrors = flockErrors
    index!
  end

  def flockDataKey
    "#{experimentNumber}.#{id}"
  end

  def FlockData.values(key)
    redisKey = "FlockData:values:#{key}"
    $redis.hgetall(redisKey)
    # $redis.keys("*")
  end
end

















require 'sequel'

class SimulationDataStoreManager
  attr_accessor :db, :exampleDataSet, :epochDataSet, :tagDataSet, :exampleFeatureDataSet,
                :theNumberOfTheCurrentExperiment, :experimentDescriptionDataSet,
                :endOfTrainingResultsDataSet, :epochNumber
  private_class_method(:new)

  @@dataStoreManager = nil

  def SimulationDataStoreManager.create(filenameOfDatabase, examples, simulationParameters)
    @@dataStoreManager = new(filenameOfDatabase, examples, simulationParameters) unless @@dataStoreManager
    @@dataStoreManager
  end

  def SimulationDataStoreManager.instance
    return @@dataStoreManager
  end

  def initialize(filenameOfDatabase, examples, simulationParameters)
    STDERR.puts "ERROR:  SimulationDataStoreManager.new called MORE THAN ONE TIME!!!!!" unless (@@dataStoreManager.nil?)
    #@db = Sequel.sqlite("../../data/acrossEpochsSequel.db")
    case filenameOfDatabase
      when ""
        @db = Sequel.sqlite # connect to an in-memory database
      else
        @db = Sequel.sqlite("../../data/#{filenameOfDatabase}.db")
        @db.pragma_set('page_size', 65536) # These 3 statements significantly speed-up performance for disk-based SQLite3 databases.
        @db.pragma_set('journal_mode', 'OFF')
        @db.pragma_set('synchronous', 'OFF')
    end

    @epochNumber = nil

    @db.create_table! :epochs do
      Bignum :epochNumber
      Integer :neuronID
      primary_key [:epochNumber, :neuronID]
      Float :wt1
      Float :wt2
      Float :cluster0Center
      Float :cluster1Center
      Float :dPrime
    end
    @epochDataSet = @db[:epochs]

    @db.create_table! :examples do
      Bignum :epochNumber
      Integer :neuronID
      Integer :exampleNumber
      primary_key [:epochNumber, :neuronID, :exampleNumber]
      Float :netInput
      Float :higherLayerError
      Float :errorToBackPropToLowerLayer
      Float :localFlockingError
      Float :weightedErrorMetric
    end
    @exampleDataSet = @db[:examples]

    @db.create_table! :exampleFeatures do
      Integer :exampleNumber, :primary_key => true
      Integer :class
      Integer :feature1
      Integer :feature2
    end
    @exampleFeatureDataSet = @db[:exampleFeatures]
    examples.each do |anExample|
      exampleFeatureDataSet.insert(:exampleNumber => anExample[:exampleNumber], :class => anExample[:class])
    end

    @db.create_table! :tags do
      Bignum :epochNumber, :primary_key => true
      Integer :experimentNumber
      Float :mse
      String :learningPhase
      Integer :epochsSinceBeginningOfPhase
      Integer :epochsSinceBeginningOfCycle
      String :note
    end
    @tagDataSet = @db[:tags]

    @db.create_table? :experimentDescriptions do
      primary_key :experimentNumber
      String :description, :text => true
      String :primaryTypeOfExperiment
      String :secondaryTypeOfExperiment
      String :simParameters, :text => true
      DateTime :startDateTime
    end
    @experimentDescriptionDataSet = @db[:experimentDescriptions]
    experimentDescriptionDataSet.insert(:description => simulationParameters[:descriptionOfExperiment], :simParameters => simulationParameters.to_s,
                                        :primaryTypeOfExperiment => 'code debugging and demo prototyping', :startDateTime => Time.now)
    @theNumberOfTheCurrentExperiment = experimentDescriptionDataSet.all.last[:experimentNumber]

    @db.create_table? :endOfTrainingResultsForAllExperiments do
      Integer :experimentNumber, :primary_key => true
      Integer :lastEpoch
      Float :lastTrainingMSE
      Float :lastTestingMSE
      Float :dPrimes
      Float :elapsedTime
    end

    @@dataStoreManager = self
  end

  def storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, dPrimes, elapsedTime)
    @endOfTrainingResultsDataSet = @db[:endOfTrainingResultsForAllExperiments]
    dPrimes = dPrimes[0] if (dPrimes.present?)
    endOfTrainingResultsDataSet.insert(:experimentNumber => theNumberOfTheCurrentExperiment,
                                       :lastEpoch => lastEpoch, :lastTrainingMSE => lastTrainingMSE,
                                       :lastTestingMSE => lastTestingMSE, :dPrimes => dPrimes,
                                       :elapsedTime => elapsedTime)
  end

  def joinDataSets
    aJoinedDS = db["SELECT neuronID, epochNumber, exampleNumber, netInput, higherLayerError, class, wt1, wt2,
 learningPhase, epochsSinceBeginningOfPhase, cluster0Center, cluster1Center, dPrime  FROM examples NATURAL JOIN
exampleFeatures NATURAL JOIN epochs NATURAL JOIN tags ORDER BY neuronID ASC, epochNumber ASC, exampleNumber ASC"]
    return aJoinedDS
  end

  def joinForShoesDisplay
    aShoesDS = db["SELECT neuronID, epochNumber, exampleNumber, netInput, higherLayerError, class, wt1, wt2  FROM examples NATURAL JOIN
exampleFeatures NATURAL JOIN epochs NATURAL JOIN tags ORDER BY neuronID ASC, epochNumber ASC, exampleNumber ASC"]
    return aShoesDS
  end

  def transferDataSetToVisualizer(dataSetFromJoin, args)
    recordedNeuronIDs = []
    arrayForShoes = dataSetFromJoin.chunk { |row| row[:neuronID] }.collect do |neuronID, neuronAry|
      recordedNeuronIDs << neuronID
      neuronAry.chunk { |row| row[:epochNumber] }.collect do |epochNumber, epochAry|
        epochAry.collect { |aHash| aHash.values[3..9] } # .values[3..-1]
      end
    end
    hashToTransfer = {:numberOfEpochsBetweenStoringDBRecords => args[:numberOfEpochsBetweenStoringDBRecords],
                      :recordedNeuronIDs => recordedNeuronIDs, :arrayForShoes => arrayForShoes}
    open('../../data/neuronData', 'w') { |f| YAML.dump(hashToTransfer, f) }
  end
end

