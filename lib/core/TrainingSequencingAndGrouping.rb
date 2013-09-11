### VERSION "nCore"
## ../nCore/lib/core/TrainingSequencingAndGrouping.rb



class TrainingSequence
  attr_accessor :args, :epochs, :epochsInPhase1, :epochsInPhase2, :numberOfEpochsInCycle,
                :maxNumberOfEpochs, :epochsSinceBeginningOfCycle, :status,
                :stillMoreEpochs, :lastEpoch, :inPhase2, :inPhase1,
                :atStartOfCycle, :atStartOfPhase1, :atStartOfPhase2,
                :atStartOfTraining, :afterFirstEpoch, :dataStoreManager,
                :tagDataSet, :epochDataSet, :numberOfEpochsBetweenStoringDBRecords

  def initialize(args)
    @args = args
    @dataStoreManager = args[:dataStoreManager]
    @maxNumberOfEpochs = args[:maxNumEpochs]
    @numberOfEpochsBetweenStoringDBRecords = @args[:numberOfEpochsBetweenStoringDBRecords]

    @epochs = -1
    @epochsSinceBeginningOfCycle = -1
    @atStartOfTraining = true
    @afterFirstEpoch = false
    @stillMoreEpochs = true
    @lastEpoch = false
    nextEpoch
  end

  def nextEpoch
    self.epochs += 1
    self.atStartOfTraining = false if (epochs > 0)
    self.afterFirstEpoch = true unless (atStartOfTraining)

    self.epochsSinceBeginningOfCycle += 1

    self.lastEpoch = false
    self.lastEpoch = true if (epochs == (maxNumberOfEpochs - 1))

    self.stillMoreEpochs = true
    self.stillMoreEpochs = false if (epochs >= maxNumberOfEpochs)

    dataStoreManager.epochNumber = epochs
  end

  def timeToRecordData
    record = false
    return if (epochs < 0)
    record = true if ((epochs % numberOfEpochsBetweenStoringDBRecords) == 0)
    record = true if lastEpoch
    return record
  end
end

#####

class AbstractNeuronGroups
  attr_accessor :allNeuronLayers, :allNeuronsInOneArray,
                :inputLayer, :outputLayer,
                :layersWithInputLinks, :adaptingLayers, :layersWhoseClustersNeedToBeSeeded,
                :neuronsWithInputLinks, :neuronsWithInputLinksInReverseOrder,
                :adaptingNeurons, :neuronsWhoseClustersNeedToBeSeeded

  def initialize(network)
    @allNeuronLayers = network.allNeuronLayers
    nameTrainingGroups()
  end

  def setUniversalNeuronGroupNames
    self.neuronsWithInputLinks = layersWithInputLinks.flatten
    self.neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
    self.allNeuronsInOneArray = inputLayer + neuronsWithInputLinks
    self.adaptingNeurons = adaptingLayers.flatten unless (adaptingLayers.nil?)
    self.neuronsWhoseClustersNeedToBeSeeded = layersWhoseClustersNeedToBeSeeded.flatten unless (layersWhoseClustersNeedToBeSeeded.nil?)
  end
end

class NeuronGroupsSimplest < AbstractNeuronGroups
  def nameTrainingGroups
    self.inputLayer = allNeuronLayers.first
    self.outputLayer = allNeuronLayers.last
    self.layersWithInputLinks = [outputLayer]
    self.adaptingLayers = [outputLayer]
    self.layersWhoseClustersNeedToBeSeeded = [outputLayer]
    setUniversalNeuronGroupNames()
  end
end

