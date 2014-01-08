### VERSION "nCore"
## ../nCore/lib/core/TrainingSequencingAndGrouping.rb

#####
class AbstractNeuronGroups
  attr_accessor :allNeuronLayers, :allNeuronsInOneArray,
                :inputLayer, :outputLayer, :hiddenLayerNeurons, :outputLayerNeurons,
                :layersWithInputLinks, :adaptingLayers,
                :neuronsWithInputLinks, :neuronsWithInputLinksInReverseOrder,
                :adaptingNeurons,
                :outputErrorGeneratingLayers, :outputErrorGeneratingNeurons,
                :outputErrorAdaptingLayers,
                :outputErrorAdaptingNeurons


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
    self.outputErrorGeneratingNeurons = (outputErrorGeneratingLayers || []).flatten
    self.outputErrorAdaptingNeurons = (outputErrorAdaptingLayers || []).flatten
  end
end

#####

class GroupsForThreeClass2HiddenLayersOEBP < AbstractNeuronGroups
  def nameTrainingGroups
    hiddenLayer1 = allNeuronLayers[1]
    hiddenLayer2 = allNeuronLayers[2]

    self.layersWithInputLinks = [hiddenLayer1, hiddenLayer2, outputLayer]
    self.outputErrorAdaptingLayers = [hiddenLayer2, outputLayer]
    setNeuronGroupNames()
  end
end


class NeuronGroupsFor3LayerBPNetwork < AbstractNeuronGroups
  def nameTrainingGroups
    hiddenLayer = allNeuronLayers[1]
    self.layersWithInputLinks = [hiddenLayer, outputLayer]
    self.outputErrorAdaptingLayers = layersWithInputLinks
    setNeuronGroupNames()
  end
end

class NeuronGroupsFor3LayerBPNetworkModLR < NeuronGroupsFor3LayerBPNetwork

  def nameTrainingGroups
    self.outputLayerNeurons = allNeuronLayers.last
    self.hiddenLayerNeurons = allNeuronLayers[1]
    super()
  end
end

class NeuronGroupsFor1LayerBPNetwork < AbstractNeuronGroups
  def nameTrainingGroups
    self.layersWithInputLinks = [outputLayer]
    self.outputErrorAdaptingLayers = layersWithInputLinks
    setNeuronGroupNames()
  end
end



#####
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



