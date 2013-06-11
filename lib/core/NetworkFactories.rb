### VERSION "nCore"
## ../nCore/lib/core/NetworkFactories.rb

require_relative 'Utilities'

############################################################      N
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

############################################################      N
class LearningNetwork
  attr_accessor :dataStoreManager, :args, :allNeuronLayers, :theBiasNeuron, :mse,
                :allNeuronsInOneArray, :inputLayer, :hiddenLayer, :outputLayer,
                :hiddenLayer1, :hiddenLayer2, :hiddenLayer3, :allHiddenLayers,
                :neuronsWithInputLinks, :neuronsWithInputLinksInReverseOrder,
                :numberOfExamples, :epochNumber,
                :networkMeanSquaredError, :networkRecorder

  def initialize(dataStoreManager, args)
    @dataStoreManager = dataStoreManager
    @args = args
    @mse = nil
    @allNeuronLayers = []
    NeuronBase.zeroID
    @theBiasNeuron = BiasNeuron.new(args)
    NeuronBase.zeroID
    @numberOfExamples = @args[:numberOfExamples]
  end

  def createSimpleLearningANN
    @networkRecorder = NetworkRecorder.new(self, args)
    createAllLayersOfNeurons()
    connectAllNeuronsToBiasNeuronExceptForThe(inputLayer)
    return allNeuronLayers
  end

  def calcNetworksMeanSquareError
    outputLayer = allNeuronLayers.last
    sse = outputLayer.inject(0.0) { |sum, anOutputNeuron| sum + anOutputNeuron.calcSumOfSquaredErrors }
    numberOfOutputNeurons = outputLayer.length
    self.mse = (sse / (numberOfOutputNeurons * numberOfExamples))
  end

  # Secondary Importance:

  def recordResponses
    networkRecorder.recordResponses
  end

  def measures
    networkRecorder.measures
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

  #def createAllLayersOfNeurons
  #  self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
  #  self.allNeuronLayers << inputLayer
  #
  #  numberOfHiddenNeurons = args[:numberOfHiddenNeurons]
  #  if (numberOfHiddenNeurons > 0)
  #    self.hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron= Neuron, numberOfHiddenNeurons)
  #    self.allNeuronLayers << hiddenLayer1
  #  end
  #
  #  self.outputLayer = createAndConnectLayer(hiddenLayer1, typeOfNeuron = OutputNeuron, args[:numberOfOutputNeurons])
  #  self.allNeuronLayers << outputLayer
  #  return allNeuronLayers
  #end

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

class SimpleFlockingNeuronNetwork < LearningNetwork

  def createAllLayersOfNeurons
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    self.outputLayer = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer

    self.allNeuronsInOneArray = allNeuronLayers.flatten
    self.neuronsWithInputLinks = outputLayer
    self.neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse

    return allNeuronLayers
  end

end # Used for main: "SimplestFlockingDemo2.rb"

class AnalogyNetwork < LearningNetwork

  def createAllLayersOfNeurons
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    self.hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron= FlockingNeuron, args[:layer1NumberOfHiddenNeurons])
    self.allNeuronLayers << hiddenLayer1

    self.hiddenLayer2 = createAndConnectLayer((inputLayer + hiddenLayer1), typeOfNeuron= FlockingNeuron, args[:layer2NumberOfHiddenNeurons])
    self.allNeuronLayers << hiddenLayer2

    self.hiddenLayer3 = createAndConnectLayer((inputLayer + hiddenLayer2), typeOfNeuron= FlockingNeuron, args[:layer3NumberOfHiddenNeurons])
    self.allNeuronLayers << hiddenLayer3

    self.outputLayer = createAndConnectLayer(hiddenLayer1, typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer

    self.allNeuronsInOneArray = allNeuronLayers.flatten
    self.neuronsWithInputLinks = outputLayer
    self.neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse

    return allNeuronLayers
  end

  def createSimpleLearningANN
    @networkRecorder = NetworkRecorder.new(self, args)
    createAllLayersOfNeurons()
    connectAllNeuronsToBiasNeuronExceptForThe(inputLayer)
    return allNeuronLayers
  end

end # Used for main: "Analogy4Class.rb"

class AnalogyNetworkNoJumpLinks < AnalogyNetwork
  def createAllLayersOfNeurons
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    self.hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron= FlockingNeuron, args[:layer1NumberOfHiddenNeurons])
    self.allNeuronLayers << hiddenLayer1

    self.hiddenLayer2 = createAndConnectLayer(hiddenLayer1, typeOfNeuron= FlockingNeuron, args[:layer2NumberOfHiddenNeurons])
    self.allNeuronLayers << hiddenLayer2

    self.hiddenLayer3 = createAndConnectLayer(hiddenLayer2, typeOfNeuron= FlockingNeuron, args[:layer3NumberOfHiddenNeurons])
    self.allNeuronLayers << hiddenLayer3

    self.outputLayer = createAndConnectLayer(hiddenLayer1, typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
    return allNeuronLayers
  end
end # Used for a variation of main: "Analogy4Class.rb"

############################################################      N
class NetworkRecorder
  attr_accessor :network, :args, :measures

  def initialize(network, args)
    @network = network
    @args = args
    @measures = []
    @trainingSequence = nil
  end

  def trainingSequence
    @trainingSequence ||= TrainingSequence.instance
  end

  def recordResponses
    measures << {:mse => network.mse, :epochs => trainingSequence.epochs} if (trainingSequence.timeToRecordData)
  end
end

