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

  def shareWeightBetweenCorrespondingLinks(sendingLayer1, receivingLayer1, sendingLayer2, receivingLayer2)
    arrayOfLinks1 = retrieveLinksBetweenGroupsOfNeurons(sendingLayer1, receivingLayer1)
    arrayOfLinks2 = retrieveLinksBetweenGroupsOfNeurons(sendingLayer2, receivingLayer2)
    STDERR.puts "Error: Number of links in the 2 groups are unequal." unless (arrayOfLinks1.length == arrayOfLinks2.length)
    arrayOfLinkArrays = arrayOfLinks1.zip(arrayOfLinks2)
    giveEachLinkArrayASingleSharedWeight(arrayOfLinkArrays)
  end

  #def shareWeightsBetweenNGroups(sendingLayer, receivingLayer, numberOfGroups)  # TODO not sure this will be correct in all use cases
  #  lengthOfReceivingLayer = receivingLayer.length
  #  STDERR.puts "Error: Number of neurons in receiving layer does not divide evenly by #{numberOfGroups}" unless ((lengthOfReceivingLayer % numberOfGroups) == 0)
  #  sliceSize = lengthOfReceivingLayer / numberOfGroups
  #
  #  arraysOfLinksToShareWeights = []
  #  receivingLayer.each_slice(sliceSize) { |partOfReceivingLayer| arraysOfLinksToShareWeights << retrieveLinksBetweenGroupsOfNeurons(sendingLayer, partOfReceivingLayer) }
  #  groupedLinks = arraysOfLinksToShareWeights.pop.zip(arraysOfLinksToShareWeights.flatten)
  #
  #  giveEachLinkArrayASingleSharedWeight(groupedLinks)
  #end
  #
  #def shareWeightsAmongNeuronsInAGroup(sendingLayer, receivingLayer, numberOfNeuronsInEachGroup)
  #  STDERR.puts "Error: Number of neurons in receiving layer does not divide evenly by #{numberOfNeuronsInEachGroup}" unless ((receivingLayer.length % numberOfNeuronsInEachGroup) == 0)
  #  receivingLayer.each_slice(numberOfNeuronsInEachGroup) do |aGroupOfReceivingNeurons|
  #    shareWeightsBetweenNGroups(sendingLayer, aGroupOfReceivingNeurons, numberOfNeuronsInEachGroup)
  #  end
  #end

  def deleteRecurrentSelfConnections(sendingLayerNeurons, receivingLayerNeurons)
    sendingLayerNeurons.each_with_index do |aSendingLayerNeuron, indexToNeuron|
      deleteCommonLinkBetweenNeurons(aSendingLayerNeuron, receivingLayerNeurons[indexToNeuron])
    end
  end

  def zeroWeightsConnecting(sendingLayerNeurons, receivingLayerNeurons)
    sendingLayerNeurons.each_with_index do |aSendingLayerNeuron, indexToNeuron|
      zeroWeightInLinkBetweenNeurons(aSendingLayerNeuron, receivingLayerNeurons[indexToNeuron])
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
    theCommonLink = findTheConnectingLink(inputLinks, outputLinks)
    outputLinks.delete(theCommonLink)
    inputLinks.delete(theCommonLink)
  end

  def findTheConnectingLink(inputLinks, outputLinks)
    theCommonLink = outputLinks.find do |anOutputLink|
      inputLinks.find { |anInputLink| anInputLink == anOutputLink }
    end
  end

  def zeroWeightsInLinksBetweenNeurons(sendingNeurons, receivingNeurons)
    sendingNeurons.each do |aSendingNeuron|
      receivingNeurons.each do |aReceivingNeuron|
        outputLinks = aSendingNeuron.outputLinks
        inputLinks = aReceivingNeuron.inputLinks
        zeroWeightInCommonLink(outputLinks, inputLinks)
      end
    end
  end

  def zeroWeightInCommonLink(outputLinks, inputLinks)
    theCommonLink = findTheConnectingLink(inputLinks, outputLinks)
    puts "Value NIL............................................" if (theCommonLink.nil?)
    theCommonLink.weight = 0.0 unless (theCommonLink.nil?)
  end

  def giveEachLinkArrayASingleSharedWeight(groupedLinks)
    groupedLinks.each do |aGroupOfLinksToShareASingleWeight|
      firstLinkOfGroup = aGroupOfLinksToShareASingleWeight[0]
      aSharedWeight = SharedWeight.new(firstLinkOfGroup.weight)
      aGroupOfLinksToShareASingleWeight.each do |aLink|
        aLink.weight = aSharedWeight
      end
    end
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

class Recurrent2HiddenLayerNetworkSpecial < BaseNetwork

  def createAllLayersOfNeurons
    STDERR.puts "Error: number of neurons in hidden layers are not identical" if (args[:numberOfHiddenLayer1Neurons] != args[:numberOfHiddenLayer2Neurons])

    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    hiddenLayer1Neurons = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingNeuron, args[:numberOfHiddenLayer1Neurons])
    self.allNeuronLayers << hiddenLayer1Neurons

    hiddenLayer2Neurons = createAndConnectLayer((inputLayer + hiddenLayer1Neurons), typeOfNeuron = FlockingNeuron, args[:numberOfHiddenLayer2Neurons])
    self.allNeuronLayers << hiddenLayer2Neurons

    # Set inter-hidden-layer weights to zero
    zeroWeightInCommonLink(hiddenLayer1Neurons, hiddenLayer2Neurons)

    # Weight Sharing code
    shareWeightBetweenCorrespondingLinks(inputLayer, hiddenLayer1Neurons, inputLayer, hiddenLayer2Neurons)

    # to create just cross-connections between 2 hidden layers of a "simulated recurrent net" we need to delete ALL (direct recurrent: N1out to N1in connections)
    deleteRecurrentSelfConnections(hiddenLayer1Neurons, hiddenLayer2Neurons)


    self.outputLayer = createAndConnectLayer((hiddenLayer1Neurons + hiddenLayer2Neurons), typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end

class Standard3LayerNetwork < BaseNetwork

  def createAllLayersOfNeurons
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    hiddenLayer = createAndConnectLayer(inputLayer, typeOfNeuron = Neuron, args[:numberOfHiddenNeurons])
    self.allNeuronLayers << hiddenLayer

    self.outputLayer = createAndConnectLayer(hiddenLayer, typeOfNeuron = OutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end

class SimpleFlockingNeuronNetwork < BaseNetwork # TODO this is identical, except in name, to  SimpleFlockingNetwork

  def createAllLayersOfNeurons
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    self.outputLayer = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end

class Flocking1LayerNetwork < BaseNetwork

  def createAllLayersOfNeurons
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    self.outputLayer = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end

class Flocking3LayerNetwork < BaseNetwork

  def createAllLayersOfNeurons
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    hiddenLayer = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingNeuron, args[:numberOfHiddenNeurons])
    self.allNeuronLayers << hiddenLayer

    self.outputLayer = createAndConnectLayer(hiddenLayer, typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end


#class DeepRecurrentNetwork < BaseNetwork  # TODO a number of things wrong here.  Needs to be corrected.
#
#  def createAllLayersOfNeurons
#    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
#    self.allNeuronLayers << inputLayer
#    previousLayer = inputLayer
#
#    args[:numberOfHiddenLayers].times do
#      hiddenLayer = createAndConnectLayer(previousLayer, typeOfNeuron = FlockingNeuron, args[:numberOfHiddenNeurons])
#      self.allNeuronLayers << hiddenLayer
#      previousLayer = hiddenLayer
#    end
#
#    self.outputLayer = createAndConnectLayer(hiddenLayer, typeOfNeuron = LinearOutputNeuron, args[:numberOfOutputNeurons])
#    self.allNeuronLayers << outputLayer
#  end
#end
#




