### VERSION "nCore"
## ../nCore/lib/core/NetworkFactories.rb

require_relative 'Utilities'

module NeuronToNeuronConnection
  def connect_layer_to_another(sendingLayer, receivingLayer, args)
    sendingLayer.each { |sendingNeuron| connectToAllNeuronsInReceivingLayer(sendingNeuron, receivingLayer, args) }
  end

  def createArrayOfNeurons(typeOfNeuron, numberOfNeurons, args)
    Array.new(numberOfNeurons) { |i| typeOfNeuron.new(args) }
  end

  def retrieveLinksBetweenGroupsOfNeurons(sendingLayer, receivingLayer)
    arrayOfCommonLinks = sendingLayer.collect do |aSendingNeuron|
      receivingLayer.collect do |aReceivingNeuron|
        retrieveLinkBetween(aSendingNeuron, aReceivingNeuron)
      end
    end
    return arrayOfCommonLinks.flatten.compact
  end

  def disconnect_one_layer_from_another(sendingLayer, receivingLayer)
    sendingLayer.each do |aSendingNeuron|
      receivingLayer.each do |aReceivingNeuron|
        deleteCommonLinkBetweenNeurons(aSendingNeuron, aReceivingNeuron)
      end
    end
  end

  private

  def connectToAllNeuronsInReceivingLayer(sendingNeuron, receivingLayer, args)
    receivingLayer.each { |receivingNeuron| connect_neuron_to_neuron(sendingNeuron, receivingNeuron, args) }
  end

  def connect_neuron_to_neuron(inputNeuron, outputNeuron, args)
    theLink = createLink(inputNeuron, outputNeuron, args)
    inputNeuron.outputLinks << theLink
    outputNeuron.inputLinks << theLink
  end

  def createLink(inputNeuron, outputNeuron, args)
    typeOfLink = args[:typeOfLink] || Link
    return typeOfLink.new(inputNeuron, outputNeuron, args)
  end

  def retrieveLinkBetween(aSendingNeuron, aReceivingNeuron)
    outputLinks = aSendingNeuron.outputLinks
    inputLinks = aReceivingNeuron.inputLinks
    theCommonLink = findCommonLink(outputLinks, inputLinks)
  end

  def findCommonLink(outputLinks, inputLinks)
    theCommonLink = outputLinks.find do |anOutputLink|
      inputLinks.find { |anInputLink| anInputLink == anOutputLink }
    end
  end

  def deleteCommonLinkBetweenNeurons(aSendingNeuron, aReceivingNeuron)
    outputLinks = aSendingNeuron.outputLinks
    inputLinks = aReceivingNeuron.inputLinks
    deleteCommonLink(outputLinks, inputLinks)
  end

  def deleteCommonLink(outputLinks, inputLinks)
    theCommonLink = outputLinks.find do |anOutputLink|
      inputLinks.find { |anInputLink| anInputLink == anOutputLink }
    end
    outputLinks.delete(theCommonLink)
    inputLinks.delete(theCommonLink)
  end
end

############################################################
class BaseNetwork
  attr_accessor :args, :allNeuronLayers, :theBiasNeuron,
                :inputLayer, :outputLayer

  def initialize(args)
    @args = args
    @allNeuronLayers = []
    NeuronBase.zeroID
    @theBiasNeuron = BiasNeuron.new(args)
    NeuronBase.zeroID
    createSimpleLearningANN
  end

  def createSimpleLearningANN
    createAllLayersOfNeurons()
    connectAllNeuronsToBiasNeuronExceptForThe(inputLayer)
  end

  def to_s
    description = "Neural Network Architecture and Parameters:"
    allNeuronLayers.each_with_index do |aLayer, index|
      description += "\n\nLayer #{index}\n"
      aLayer.each { |aNeuron| description += aNeuron.to_s }
    end
    return description
  end

  protected

  include NeuronToNeuronConnection

  def createAndConnectLayer(inputToLayer, typeOfNeuronInLayer, numberOfNeurons)
    layer = createArrayOfNeurons(typeOfNeuronInLayer, numberOfNeurons, args)
    connect_layer_to_another(inputToLayer, layer, args) unless (inputToLayer.nil?) # input neurons do not receive any inputs from other neurons
    return layer
  end

  def connectAllNeuronsToBiasNeuronExceptForThe(inputNeurons)
    addLinksFromBiasNeuronToHiddenAndOutputNeurons(allNeuronLayers.flatten - inputNeurons)
  end

  def addLinksFromBiasNeuronToHiddenAndOutputNeurons(singleArrayOfAllNeuronsToReceiveBiasInput)
    connect_layer_to_another([theBiasNeuron], singleArrayOfAllNeuronsToReceiveBiasInput, args)
  end
end # Base network

class SimpleFlockingNeuronNetwork < BaseNetwork   # TODO this is identical, except in name, to  SimpleFlockingNetwork

  def createAllLayersOfNeurons
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    self.outputLayer = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end

class SimpleFlockingNetwork < BaseNetwork

  def createAllLayersOfNeurons
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    self.outputLayer = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end

class BPofFlockingNetwork < BaseNetwork

  def createAllLayersOfNeurons
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    hiddenLayer = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingNeuron, args[:numberOfHiddenNeurons])
    self.allNeuronLayers << hiddenLayer

    self.outputLayer = createAndConnectLayer(hiddenLayer, typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end



