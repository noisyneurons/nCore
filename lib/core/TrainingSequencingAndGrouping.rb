### VERSION "nCore"
## ../nCore/lib/core/TrainingSequencingAndGrouping.rb

class TrainingSequence
  attr_accessor :args, :epochs, :maxNumberOfEpochs,
                :stillMoreEpochs, :lastEpoch,
                :atStartOfTraining, :afterFirstEpoch

  def initialize(args)
    @args = args
    @maxNumberOfEpochs = args[:maxNumEpochs]
    @epochs = -1
    @atStartOfTraining = true
    @afterFirstEpoch = false
    @stillMoreEpochs = true
    @lastEpoch = false
    nextEpoch
  end

  def epochs=(value)
    @epochs = value
    args[:epochs] = value
  end

  def nextEpoch
    self.epochs += 1
    self.atStartOfTraining = false if (epochs > 0)
    self.afterFirstEpoch = true unless (atStartOfTraining)

    self.lastEpoch = false
    self.lastEpoch = true if (epochs == (maxNumberOfEpochs - 1))

    self.stillMoreEpochs = true
    self.stillMoreEpochs = false if (epochs >= maxNumberOfEpochs)
  end
end


#####
class AbstractNeuronGroups
  attr_accessor :allNeuronLayers, :allNeuronsInOneArray,
                :inputLayer, :outputLayer,
                :layersWithInputLinks, :adaptingLayers, :layersWhoseClustersNeedToBeSeeded,
                :neuronsWithInputLinks, :neuronsWithInputLinksInReverseOrder,
                :adaptingNeurons, :neuronsWhoseClustersNeedToBeSeeded,
                :outputErrorGeneratingLayers, :outputErrorGeneratingNeurons

  def initialize(network)
    @allNeuronLayers = network.allNeuronLayers
    @inputLayer = allNeuronLayers.first
    @outputLayer = allNeuronLayers.last
    @outputErrorGeneratingLayers = [outputLayer]
    nameTrainingGroups()
  end

  def setNeuronGroupNames
    self.neuronsWithInputLinks = layersWithInputLinks.flatten
    self.neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
    self.allNeuronsInOneArray = inputLayer + neuronsWithInputLinks
    self.neuronsWhoseClustersNeedToBeSeeded  = layersWhoseClustersNeedToBeSeeded.flatten unless (layersWhoseClustersNeedToBeSeeded.nil?)
    self.outputErrorGeneratingNeurons = outputErrorGeneratingLayers.flatten unless (outputErrorGeneratingLayers.nil?)
  end
end


#####
class NeuronGroupsSimplest < AbstractNeuronGroups
  def nameTrainingGroups
    self.layersWithInputLinks = [outputLayer]
    self.adaptingLayers = [outputLayer]
    self.layersWhoseClustersNeedToBeSeeded = [outputLayer]
    setNeuronGroupNames()
  end

  def setNeuronGroupNames
    super
    self.adaptingNeurons = adaptingLayers.flatten unless (adaptingLayers.nil?)
  end
end


#####
class NeuronGroupsTrivial < AbstractNeuronGroups
  attr_accessor :outputErrorAdaptingLayers,:flockErrorGeneratingLayers,:flockErrorAdaptingLayers,
                :outputErrorAdaptingNeurons,:flockErrorGeneratingNeurons,:flockErrorAdaptingNeurons
  def nameTrainingGroups
    self.layersWithInputLinks = [outputLayer]

    self.outputErrorAdaptingLayers = [outputLayer]
    self.flockErrorGeneratingLayers = [outputLayer]
    self.flockErrorAdaptingLayers =  [outputLayer]

    self.layersWhoseClustersNeedToBeSeeded = flockErrorGeneratingLayers
    setNeuronGroupNames()
  end

  def setNeuronGroupNames
    super
    self.outputErrorAdaptingNeurons = outputErrorAdaptingLayers.flatten unless (outputErrorAdaptingLayers.nil?)
    self.flockErrorGeneratingNeurons = flockErrorGeneratingLayers.flatten unless (flockErrorGeneratingLayers.nil?)
    self.flockErrorAdaptingNeurons = flockErrorAdaptingLayers.flatten unless (flockErrorAdaptingLayers.nil?)
  end
end


#####
class NeuronGroups1BPofFlocking < NeuronGroupsTrivial
  attr_accessor :layersToReceive_backPropagate_message, :neuronsToReceive_backPropagate_message
  def nameTrainingGroups
    self.layersWithInputLinks = [outputLayer]
    self.outputErrorAdaptingLayers = [outputLayer]
    self.flockErrorGeneratingLayers = [outputLayer]
    self.flockErrorAdaptingLayers =  [outputLayer]
    self.layersWhoseClustersNeedToBeSeeded = flockErrorGeneratingLayers

    self.layersToReceive_backPropagate_message = nil

    setNeuronGroupNames()
  end

  def setNeuronGroupNames
    super
    neuronsToReceive_backPropagate_message = layersToReceive_backPropagate_message.flatten unless (layersToReceive_backPropagate_message.nil?)
  end
end


#class NeuronGroups2BPofFlocking < AbstractNeuronGroups
#  def nameTrainingGroups
#    self.inputLayer = allNeuronLayers.first
#    hiddenLayer = allNeuronLayers[1]
#    self.outputLayer = allNeuronLayers.last
#    self.layersWithInputLinks = [hiddenLayer, outputLayer]
#
#    self.outputErrorGeneratingLayers = [outputLayer]
#    self.outputErrorReceivingLayers = [hiddenLayer, outputLayer]
#    self.flockErrorGeneratingLayers = [outputLayer]
#    self.flockErrorReceivingLayers =  [outputLayer]
#
#    self.layersWhoseClustersNeedToBeSeeded = flockErrorGeneratingLayers
#    setNeuronGroupNames()
#  end
#end
#
