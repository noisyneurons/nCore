### VERSION "nCore"
## ../nCore/lib/core/TrainingSequencingAndGrouping.rb

#####
class AbstractNeuronGroups
  attr_accessor :allNeuronLayers, :allNeuronsInOneArray,
                :inputLayer, :outputLayer,
                :layersWithInputLinks, :adaptingLayers, :layersWhoseClustersNeedToBeSeeded,
                :neuronsWithInputLinks, :neuronsWithInputLinksInReverseOrder,
                :adaptingNeurons, :neuronsWhoseClustersNeedToBeSeeded,
                :outputErrorGeneratingLayers, :outputErrorGeneratingNeurons,
                :outputErrorAdaptingLayers, :flockErrorGeneratingLayers, :flockErrorAdaptingLayers,
                :outputErrorAdaptingNeurons, :flockErrorGeneratingNeurons, :flockErrorAdaptingNeurons,
                :bpFlockErrorAdaptingNeurons, :bpFlockErrorAdaptingLayers,
                :bpFlockErrorGeneratingNeurons, :bpFlockErrorGeneratingLayers


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
    self.neuronsWhoseClustersNeedToBeSeeded = layersWhoseClustersNeedToBeSeeded.flatten
    self.outputErrorGeneratingNeurons = (outputErrorGeneratingLayers || []).flatten
    self.outputErrorAdaptingNeurons = (outputErrorAdaptingLayers || []).flatten
    self.flockErrorGeneratingNeurons = (flockErrorGeneratingLayers || []).flatten
    self.flockErrorAdaptingNeurons = (flockErrorAdaptingLayers || []).flatten
    self.bpFlockErrorAdaptingNeurons = (bpFlockErrorAdaptingLayers || []).flatten
    self.bpFlockErrorGeneratingNeurons = (bpFlockErrorGeneratingLayers || []).flatten
  end
end


#####
#class GroupsForThreeClass2HiddenLayersOEBP < AbstractNeuronGroups
#  def nameTrainingGroups
#    hiddenLayer1 = allNeuronLayers[1]
#
#    self.layersWithInputLinks = [hiddenLayer1, outputLayer]
#
#    self.outputErrorAdaptingLayers = layersWithInputLinks
#    self.flockErrorGeneratingLayers = []
#    self.flockErrorAdaptingLayers = []
#
#    self.layersWhoseClustersNeedToBeSeeded = flockErrorGeneratingLayers
#    setNeuronGroupNames()
#  end
#end


class GroupsForThreeClass2HiddenLayersOEBP < AbstractNeuronGroups
  def nameTrainingGroups
    hiddenLayer1 = allNeuronLayers[1]
    hiddenLayer2 = allNeuronLayers[2]

    self.layersWithInputLinks = [hiddenLayer1, hiddenLayer2, outputLayer]

    self.outputErrorAdaptingLayers = [hiddenLayer2, outputLayer]
    self.flockErrorGeneratingLayers = []
    self.flockErrorAdaptingLayers = []

    self.layersWhoseClustersNeedToBeSeeded = flockErrorGeneratingLayers
    setNeuronGroupNames()
  end
end


class NeuronGroupsFor3LayerBPNetwork < AbstractNeuronGroups
  def nameTrainingGroups
    hiddenLayer = allNeuronLayers[1]
    self.layersWithInputLinks = [hiddenLayer, outputLayer]
    self.outputErrorAdaptingLayers = layersWithInputLinks
    self.layersWhoseClustersNeedToBeSeeded = []
    setNeuronGroupNames()
  end
end


class NeuronGroupsForSingleLayerNetwork < AbstractNeuronGroups

  def nameTrainingGroups
    self.layersWithInputLinks = [outputLayer]

    self.outputErrorAdaptingLayers = [outputLayer]
    self.flockErrorGeneratingLayers = [outputLayer]
    self.flockErrorAdaptingLayers = [outputLayer]

    self.layersWhoseClustersNeedToBeSeeded = flockErrorGeneratingLayers
    setNeuronGroupNames()
  end
end


class NeuronGroupsHiddenLayerLocalFlocking < NeuronGroupsForSingleLayerNetwork
  def nameTrainingGroups
    hiddenLayer = allNeuronLayers[1]
    self.layersWithInputLinks = [hiddenLayer, outputLayer]

    self.outputErrorAdaptingLayers = layersWithInputLinks
    self.flockErrorGeneratingLayers = [hiddenLayer]
    self.flockErrorAdaptingLayers = [hiddenLayer]

    self.layersWhoseClustersNeedToBeSeeded = flockErrorGeneratingLayers
    setNeuronGroupNames()
  end
end


class GroupsForThreeClass2HiddenLayers < AbstractNeuronGroups
  def nameTrainingGroups
    hiddenLayer1 = allNeuronLayers[1]
    hiddenLayer2 = allNeuronLayers[2]

    self.layersWithInputLinks = [hiddenLayer1, hiddenLayer2, outputLayer]

    self.outputErrorAdaptingLayers = layersWithInputLinks
    self.flockErrorGeneratingLayers = [hiddenLayer2]
    self.flockErrorAdaptingLayers = [hiddenLayer2]

    self.layersWhoseClustersNeedToBeSeeded = flockErrorGeneratingLayers
    setNeuronGroupNames()
  end
end


class NeuronGroups3LayersAllLocalFlockingLayers < NeuronGroupsForSingleLayerNetwork
  def nameTrainingGroups
    hiddenLayer = allNeuronLayers[1]
    self.layersWithInputLinks = [hiddenLayer, outputLayer]

    self.outputErrorAdaptingLayers = layersWithInputLinks
    self.flockErrorGeneratingLayers = layersWithInputLinks
    self.flockErrorAdaptingLayers = layersWithInputLinks

    self.layersWhoseClustersNeedToBeSeeded = flockErrorGeneratingLayers
    setNeuronGroupNames()
  end
end


class NeuronGroups3LayersOutputLayerLocalFlocking < NeuronGroupsForSingleLayerNetwork
  def nameTrainingGroups
    hiddenLayer = allNeuronLayers[1]
    self.layersWithInputLinks = [hiddenLayer, outputLayer]

    self.outputErrorAdaptingLayers = layersWithInputLinks
    self.flockErrorGeneratingLayers = [outputLayer]
    self.flockErrorAdaptingLayers = [outputLayer]

    self.layersWhoseClustersNeedToBeSeeded = flockErrorGeneratingLayers
    setNeuronGroupNames()
  end
end


class NeuronGroups3LayersOutputLocalAndBPFlocking < NeuronGroupsForSingleLayerNetwork
  def nameTrainingGroups
    hiddenLayer = allNeuronLayers[1]
    self.layersWithInputLinks = [hiddenLayer, outputLayer]

    self.outputErrorAdaptingLayers = layersWithInputLinks
    self.flockErrorGeneratingLayers = [outputLayer]
    self.flockErrorAdaptingLayers = [outputLayer]
    self.bpFlockErrorAdaptingLayers = [hiddenLayer]
    self.bpFlockErrorGeneratingLayers = [outputLayer]

    self.layersWhoseClustersNeedToBeSeeded = [outputLayer]
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



