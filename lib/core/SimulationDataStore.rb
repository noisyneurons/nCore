### VERSION "nCore"
## ../nCore/lib/core/SimulationDataStore.rb

# require_relative 'Utilities'


#############################

#class ExperimentLogger
#  attr_reader :descriptionOfExperiment, :experimentNumber, :args
#
#  $redis.setnx("experimentNumber", 0)
#
#  #def deleteTemporaryDataRecordsInDB
#  #  #TrainingData.deleteData(experimentNumber)
#  #  #TrainingData.deleteEntireIndex!
#  #  NeuronData.deleteData(experimentNumber)
#  #  NeuronData.deleteEntireIndex!
#  #  DetailedNeuronData.deleteData(experimentNumber)
#  #  DetailedNeuronData.deleteEntireIndex!
#  #end
#
#  def ExperimentLogger.deleteTable!
#    $redis.del("experimentNumber")
#  end
#
#  def ExperimentLogger.initializeExperimentNumber(experimentNumber)
#    $redis.setnx("experimentNumber", experimentNumber)
#  end
#
#  def initialize(descriptionOfExperiment = nil, jobName = "NotNamed")
#    @experimentNumber = $redis.incr("experimentNumber")
#    @descriptionOfExperiment = descriptionOfExperiment
#    $redis.rpush("#{jobName}List", experimentNumber)
#  end
#
#end

#############################

#class SnapShotData
#  include Relix
#  Relix.host = $currentHost
#  attr_accessor :id, :experimentNumber, :descriptionOfExperiment, :network, :time, :epochs, :trainMSE, :testMSE
#
#  relix do
#    primary_key :dataKey
#    ordered :experimentNumber
#    multi :experimentNumber_epochs, on: %w(experimentNumber epochs)
#    # multi :experimentNumber, index_values: true
#  end
#
#  @@ID = 0
#
#  def SnapShotData.values(key)
#    YAML.load($redis.get(key))
#  end
#
#  def SnapShotData.deleteData(experimentNumber)
#    ary = $redis.keys("SSD#{experimentNumber}*")
#    ary.each { |item| $redis.del(item) }
#  end
#
#  def SnapShotData.deleteEntireIndex!
#    ary = $redis.keys("SSD*")
#    ary.each { |item| $redis.del(item) }
#  end
#
#  def SnapShotData.deleteKey(experimentNumber)
#    ary = $redis.keys("SnapShotData#{experimentNumber}*")
#    ary.each { |item| $redis.del(item) }
#  end
#
#  def initialize(dataToRecord)
#    @id = @@ID
#    @@ID += 1
#    @experimentNumber = dataToRecord[:experimentNumber]
#    @epochs = dataToRecord[:epochs]
#
#    $redis.set(dataKey, YAML.dump(dataToRecord))
#    index!
#  end
#
#  def dataKey
#    "SSD#{experimentNumber}.#{id}"
#  end
#end
#
#class DetailedNeuronData
#  include Relix
#  Relix.host = $currentHost
#  attr_accessor :id, :experimentNumber, :epochs, :neuron, :exampleNumber
#
#  relix do
#    primary_key :detailedNeuronDataKey
#    ordered :experimentNumber
#    multi :experimentNumber_epochs_neuron_exampleNumber, on: %w(experimentNumber epochs neuron exampleNumber)
#    multi :experimentNumber, index_values: true
#    multi :epochs, index_values: true
#    multi :exampleNumber, index_values: true
#  end
#
#  @@ID = 0
#
#  def DetailedNeuronData.values(key)
#    YAML.load($redis.get(key))
#  end
#
#  def DetailedNeuronData.deleteData(experimentNumber)
#    ary = $redis.keys("DND#{experimentNumber}*")
#    ary.each { |item| $redis.del(item) }
#  end
#
#  def DetailedNeuronData.deleteEntireIndex!
#    ary = $redis.keys("DetailedNeuronData*")
#    ary.each { |item| $redis.del(item) }
#  end
#
#  def initialize(detailedNeuronDataToRecord)
#    @id = @@ID
#    @@ID += 1
#    @experimentNumber = $globalExperimentNumber
#    @epochs = detailedNeuronDataToRecord[:epochs]
#    @neuron = detailedNeuronDataToRecord[:neuronID]
#    @exampleNumber = detailedNeuronDataToRecord[:exampleNumber]
#    # logger.puts "detailedNeuronDataToRecord = #{detailedNeuronDataToRecord}"
#    $redis.set(detailedNeuronDataKey, YAML.dump(detailedNeuronDataToRecord))
#    index!
#  end
#
#  def detailedNeuronDataKey
#    "DND#{experimentNumber}.#{id}"
#  end
#end
#
#class NeuronData
#  include Relix
#  Relix.host = $currentHost
#  attr_accessor :id, :experimentNumber, :epochs, :neuron
#
#  relix do
#    primary_key :neuronDataKey
#    # ordered :experimentNumber
#    multi :experimentNumber_epochs_neuron, on: %w(experimentNumber epochs neuron)
#    multi :experimentNumber, index_values: true
#    multi :epochs, index_values: true
#  end
#
#  @@ID = 0
#
#  def NeuronData.values(key)
#    YAML.load($redis.get(key))
#  end
#
#  def NeuronData.deleteData(experimentNumber)
#    ary = $redis.keys("ND#{experimentNumber}*")
#    ary.each { |item| $redis.del(item) }
#  end
#
#  def NeuronData.deleteEntireIndex!
#    ary = $redis.keys("NeuronData*")
#    ary.each { |item| $redis.del(item) }
#  end
#
#  def initialize(neuronDataToRecord)
#    @id = @@ID
#    @@ID += 1
#    @experimentNumber = $globalExperimentNumber
#    @neuron = neuronDataToRecord[:neuronID]
#    @epochs = neuronDataToRecord[:epochs]
#    $redis.set(neuronDataKey, YAML.dump(neuronDataToRecord))
#    index!
#  end
#
#  def neuronDataKey
#    "ND#{experimentNumber}.#{id}"
#  end
#end
#
#class TrainingData
#  include Relix
#  Relix.host = $currentHost
#  attr_accessor :id, :experimentNumber, :epochs
#
#  relix do
#    primary_key :trainingDataKey
#    multi :experimentNumber_epochs, on: %w(experimentNumber epochs)
#    multi :experimentNumber, index_values: true
#    multi :epochs, index_values: true
#  end
#
#  @@ID = 0
#
#  def TrainingData.values(key)
#    YAML.load($redis.get(key))
#  end
#
#  def TrainingData.deleteData(experimentNumber)
#    ary = $redis.keys("TD#{experimentNumber}*")
#    ary.each { |item| $redis.del(item) }
#  end
#
#  def TrainingData.deleteEntireIndex!
#    ary = $redis.keys("TrainingData*")
#    ary.each { |item| $redis.del(item) }
#  end
#
#  def initialize(trainingDataToRecord)
#    @id = @@ID
#    @@ID += 1
#    @experimentNumber = $globalExperimentNumber
#    @epochs = trainingDataToRecord[:epochs]
#    $redis.set(trainingDataKey, YAML.dump(trainingDataToRecord))
#    index!
#  end
#
#  def trainingDataKey
#    "TD#{experimentNumber}.#{id}"
#  end
#end

#####################################

#module DBAccess
#  def dbStoreNeuronData
#    savingInterval = args[:intervalForSavingNeuronData]
#    if recordOrNot?(savingInterval)
#      aHash = metricRecorder.dataToRecord
#      aHash.delete(:exampleNumber)
#      aHash.delete(:error)
#      aHash[:epochs] = args[:epochs]
#      # logger.puts "stored NeuronData #{aHash}"
#      NeuronData.new(aHash)
#    end
#  end
#
#  def dbStoreDetailedData
#    savingInterval = args[:intervalForSavingDetailedNeuronData]
#    if recordOrNot?(savingInterval)
#      aHash = metricRecorder.dataToRecord
#      aHash.delete(:error)
#      aHash[:epochs] = args[:epochs]
#      # logger.puts "----> stored DetailedNeuronData #{aHash}"
#      DetailedNeuronData.new(aHash)
#    end
#  end
#
#  def dbStoreTrainingData
#    savingInterval = args[:intervalForSavingTrainingData]
#    if recordOrNot?(savingInterval)
#      trainMSE = calcMeanSumSquaredErrors
#      testMSE = calcTestingMeanSquaredErrors
#      # theWeightsValue = outputLayer[0].inputLinks.last.weight
#      logger.puts "epoch number = #{args[:epochs]}\ttrainMSE = #{trainMSE}\ttestMSE = #{testMSE}" # "\ttheWeightsValue = #{theWeightsValue}" # unless($currentHost == "master")
#      aHash = {:experimentNumber => $globalExperimentNumber, :epochs => args[:epochs], :mse => trainMSE, :testMSE => testMSE}
#      TrainingData.new(aHash)
#    end
#  end
#
#  def recordOrNot?(recordingInterval)
#    epochs = args[:epochs]
#    return true if (epochs == 0)
#    return (((epochs + 1) % recordingInterval) == 0)
#  end
#end
#
#class Neuron
#  include DBAccess
#end
#
#class OutputNeuron
#  include DBAccess
#end
#
#class AbstractStepTrainer
#  include DBAccess
#end

#####################################

#class SimulationDataStoreManager
#  attr_accessor :args
#
#  def initialize(args={})
#    @args = args
#  end
#
#  def deleteTemporaryDataRecordsInDB(experimentNumber)
#    #TrainingData.deleteData(experimentNumber)
#    #TrainingData.deleteEntireIndex!
#    NeuronData.deleteData(experimentNumber)
#    NeuronData.deleteEntireIndex!
#    DetailedNeuronData.deleteData(experimentNumber)
#    DetailedNeuronData.deleteEntireIndex!
#  end
#
#  def deleteDataForExperiment(experimentNumber)
#    DetailedNeuronData.deleteData(experimentNumber)
#    NeuronData.deleteData(experimentNumber)
#    TrainingData.deleteData(experimentNumber)
#  end
#
#  def deleteAllDataAndIndexesExceptSnapShot!
#    nextExperimentNumber = $redis.get("experimentNumber")
#    (1...nextExperimentNumber.to_i).each do |experimentNumber|
#      deleteDataForExperiment(experimentNumber)
#    end
#    DetailedNeuronData.deleteEntireIndex!
#    NeuronData.deleteEntireIndex!
#    TrainingData.deleteEntireIndex!
#  end
#
#  def save
#    begin
#      $redis.save
#    rescue Exception
#      logger.puts "redis store was already being saved!"
#    end
#  end
#end


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
#    cluster.membershipWeightForEachExample.each { |exampleWeight| logger.puts "Epoch Number, Cluster Number and Example Weighting= #{epochDataToRecord[:epochNumber]}\t#{numberOfCluster}\t#{exampleWeight}" }
#    logger.puts
#    logger.puts "NumExamples=\t#{cluster.numExamples}\tNum Membership Weights=\t#{cluster.membershipWeightForEachExample.length}"
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


