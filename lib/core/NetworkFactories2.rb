### VERSION "nCore"
## ../nCore/lib/core/NetworkFactories2.rb

########################################################################

class SelfOrg1NeuronNetwork < BaseNetwork

  attr_accessor :hiddenLayers, :hiddenNeurons

  def createNetwork
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, typeOfLink = args[:typeOfLink], args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer1Neurons])
    self.allNeuronLayers << hiddenLayer1

    connectAllLearningNeuronsToBiasNeuron
  end

  def connectAllLearningNeuronsToBiasNeuron
    connectAllNeuronsToBiasNeuronExceptForThe(inputLayer)
  end

  def addLinksFromBiasNeuronTo(neurons, typeOfLink)
    connect_layer_to_another([theBiasNeuron], neurons, typeOfLink, args)
  end
end

class ContextNetwork < BaseNetwork
  attr_accessor :hiddenLayers, :hiddenNeurons

  def createNetwork
    createLayersWithContextLayerArchitecture
    self.hiddenLayers = allNeuronLayers[1..-2]
    self.hiddenNeurons = hiddenLayers.flatten
    connectAllLearningNeuronsToBiasNeuron
  end

  ### bew below


  def context_connect_layers(sendingLayer, receivingLayer, typeOfLink, args)
    sendingLayer.each_with_index do |sendingNeuron, index|
      receivingNeuron = receivingLayer[index * 2]
      connect_neuron_to_context_neuron(sendingNeuron, receivingNeuron, typeOfLink, args)
      receivingNeuron = receivingLayer[(index * 2) + 1]
      connect_neuron_to_context_neuron(sendingNeuron, receivingNeuron, typeOfLink, args)
    end
  end

  def connect_neuron_to_context_neuron(sendingNeuron, receivingNeuron, typeOfLink, args)
    theLink = createLink(sendingNeuron, receivingNeuron, typeOfLink, args)
    sendingNeuron.outputLinks << theLink
    receivingNeuron.suppressorLink = theLink
  end

  #### old below

  def connectAllLearningNeuronsToBiasNeuron
    addLinksFromBiasNeuronTo(hiddenNeurons, args[:typeOfLink])
    addLinksFromBiasNeuronTo(outputLayer, args[:typeOfLinkToOutput])
  end

  def addLinksFromBiasNeuronTo(neurons, typeOfLink)
    connect_layer_to_another([theBiasNeuron], neurons, typeOfLink, args)
  end
end

class Context4LayerNetworkVer1 < ContextNetwork

  def createLayersWithContextLayerArchitecture
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, typeOfLink = args[:typeOfLink], args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer1Neurons])
    self.allNeuronLayers << hiddenLayer1

    hiddenLayer2 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer2Neurons])
    self.allNeuronLayers << hiddenLayer2

    self.outputLayer = createAndConnectLayer((hiddenLayer1 + hiddenLayer2), typeOfNeuron = args[:typeOfOutputNeuron], typeOfLink = args[:typeOfLinkToOutput], args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end

class Context4LayerNetworkVer2 < ContextNetwork

  def createLayersWithContextLayerArchitecture
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, typeOfLink = args[:typeOfLink], args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer1Neurons])
    self.allNeuronLayers << hiddenLayer1

    hiddenLayer2 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer2Neurons])
    self.allNeuronLayers << hiddenLayer2

    context_connect_layers(hiddenLayer1, hiddenLayer2, SuppressorLink, args)

    self.outputLayer = createAndConnectLayer((hiddenLayer1 + hiddenLayer2), typeOfNeuron = args[:typeOfOutputNeuron], typeOfLink = args[:typeOfLinkToOutput], args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end

end


