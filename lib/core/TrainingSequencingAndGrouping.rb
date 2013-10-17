### VERSION "nCore"
## ../nCore/lib/core/TrainingSequencingAndGrouping.rb

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
    self.neuronsWhoseClustersNeedToBeSeeded = layersWhoseClustersNeedToBeSeeded.flatten unless (layersWhoseClustersNeedToBeSeeded.nil?)
    self.outputErrorGeneratingNeurons = outputErrorGeneratingLayers.flatten unless (outputErrorGeneratingLayers.nil?)
    self.outputErrorAdaptingNeurons = outputErrorAdaptingLayers.flatten unless (outputErrorAdaptingLayers.nil?)
    self.flockErrorGeneratingNeurons = flockErrorGeneratingLayers.flatten unless (flockErrorGeneratingLayers.nil?)
    self.flockErrorAdaptingNeurons = flockErrorAdaptingLayers.flatten unless (flockErrorAdaptingLayers.nil?)
    self.bpFlockErrorAdaptingNeurons = bpFlockErrorAdaptingLayers.flatten unless (bpFlockErrorAdaptingLayers.nil?)
  end
end


#####
class NeuronGroupsForSingleLayerNetwork < AbstractNeuronGroups
  attr_accessor :outputErrorAdaptingLayers, :flockErrorGeneratingLayers, :flockErrorAdaptingLayers,
                :outputErrorAdaptingNeurons, :flockErrorGeneratingNeurons, :flockErrorAdaptingNeurons,
                :bpFlockErrorAdaptingNeurons, :bpFlockErrorAdaptingLayers

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

    self.layersWhoseClustersNeedToBeSeeded = flockErrorGeneratingLayers
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



