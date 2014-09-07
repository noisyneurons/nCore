### VERSION "nCore"
## ../nCore/lib/core/NetworkFactories.rb

require_relative 'Utilities'

module NeuronToNeuronConnection
  def connect_layer_to_another(sendingLayer, receivingLayer, typeOfLink, args)
    sendingLayer.each { |sendingNeuron| connectToAllNeuronsInReceivingLayer(sendingNeuron, receivingLayer, typeOfLink, args) }
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

  def connectToAllNeuronsInReceivingLayer(sendingNeuron, receivingLayer, typeOfLink, args)
    receivingLayer.each { |receivingNeuron| connect_neuron_to_neuron(sendingNeuron, receivingNeuron, typeOfLink, args) }
  end

  def connect_neuron_to_neuron(inputNeuron, outputNeuron, typeOfLink, args)
    theLink = createLink(inputNeuron, outputNeuron, typeOfLink, args)
    inputNeuron.outputLinks << theLink
    outputNeuron.inputLinks << theLink
  end

  def createLink(inputNeuron, outputNeuron, typeOfLink, args)
    typeOfLink = typeOfLink
    return typeOfLink.new(inputNeuron, outputNeuron, args)
  end

  def retrieveLinkBetween(aSendingNeuron, aReceivingNeuron)
    outputLinks = aSendingNeuron.outputLinks
    inputLinks = aReceivingNeuron.inputLinks
    theCommonLink = findTheConnectingLink(inputLinks, outputLinks)
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
    if (theCommonLink.nil?)
      STDERR.puts "Possible ERROR: No common link between 2 Neurons"
    else
      theCommonLink.weight = 0.0
    end
  end

  def zeroLearningRateInLinksBetweenNeurons(sendingNeurons, receivingNeurons)
    sendingNeurons.each do |aSendingNeuron|
      receivingNeurons.each do |aReceivingNeuron|
        outputLinks = aSendingNeuron.outputLinks
        inputLinks = aReceivingNeuron.inputLinks
        zeroLearningRateInCommonLink(outputLinks, inputLinks)
      end
    end
  end

  def zeroLearningRateInCommonLink(outputLinks, inputLinks)
    theCommonLink = findTheConnectingLink(inputLinks, outputLinks)
    if (theCommonLink.nil?)
      STDERR.puts "Possible ERROR: No common link between 2 Neurons"
    else
      theCommonLink.learningRate = 0.0
    end
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
  include NeuronToNeuronConnection

  def initialize(args)
    @args = args
    @allNeuronLayers = []
    NeuronBase.zeroID
    @theBiasNeuron = BiasNeuron.new(args)
    createNetwork
  end

  def to_s
    description = "Neural Network Architecture and Parameters:\n"
    description += "#{theBiasNeuron}\n"
    allNeuronLayers.each_with_index do |aLayer, index|
      description += "\n\nLayer #{index}\n"
      aLayer.each { |aNeuron| description += aNeuron.to_s }
    end
    return description
  end

  protected

  def createAndConnectLayer(inputToLayer, typeOfNeuronInLayer, typeOfLink, numberOfNeurons)
    layer = createArrayOfNeurons(typeOfNeuronInLayer, numberOfNeurons, args)
    connect_layer_to_another(inputToLayer, layer, typeOfLink, args) unless (inputToLayer.nil?) # input neurons do not receive any inputs from other neurons
    return layer
  end

  def connectAllNeuronsToBiasNeuronExceptForThe(inputNeurons)
    addLinksFromBiasNeuronToHiddenAndOutputNeurons(allNeuronLayers.flatten - inputNeurons)
  end

  def addLinksFromBiasNeuronToHiddenAndOutputNeurons(singleArrayOfAllNeuronsToReceiveBiasInput)
    connect_layer_to_another([theBiasNeuron], singleArrayOfAllNeuronsToReceiveBiasInput, args[:typeOfLink], args)
  end
end # Base network


class Simplest1LayerNet < BaseNetwork

  def createNetwork
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, typeOfLink= args[:typeOfLink], args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    self.outputLayer = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfOutputNeuron], typeOfLink= args[:typeOfLink], args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer

    connectAllNeuronsToBiasNeuronExceptForThe(inputLayer)
    self
  end
end


class Standard3LayerNetwork < BaseNetwork

  def createNetwork
    createLayersAndSequentialLayerToLayerConnections
    connectAllNeuronsToBiasNeuronExceptForThe(inputLayer)
  end

  def createLayersAndSequentialLayerToLayerConnections
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, typeOfLink= args[:typeOfLink], args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    hiddenLayer = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink= args[:typeOfLink], args[:numberOfHiddenNeurons])
    self.allNeuronLayers << hiddenLayer

    self.outputLayer = createAndConnectLayer(hiddenLayer, typeOfNeuron = args[:typeOfOutputNeuron], typeOfLink= args[:typeOfLink], args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end


class JumpLinked3LayerNetwork < Standard3LayerNetwork

  def createNetwork
    createLayersAndSequentialLayerToLayerConnections
    connect_layer_to_another(inputLayer, outputLayer, args[:typeOfLink], args) # This is where we create links that 'jump across' the hidden layer.
    connectAllNeuronsToBiasNeuronExceptForThe(inputLayer)
  end

end



###########################################################################################
# May want to move this to a separate file and place it after this file in the "include manifest of files to load"

###########################################################################################

class ContextNetwork < BaseNetwork
  attr_accessor :hiddenLayers, :hiddenNeurons

  def createNetwork
    createLayersWithContextLayerArchitecture
    self.hiddenLayers = allNeuronLayers[1..-2]
    self.hiddenNeurons =  hiddenLayers.flatten
    connectAllLearningNeuronsToBiasNeuron
  end

  def connectAllLearningNeuronsToBiasNeuron
    addLinksFromBiasNeuronTo( hiddenNeurons, args[:typeOfLink] )
    addLinksFromBiasNeuronTo( outputLayer, args[:typeOfLinkToOutput] )
  end

  def addLinksFromBiasNeuronTo( neurons, typeOfLink)
    connect_layer_to_another([theBiasNeuron], neurons, typeOfLink, args)
  end
end



class Context4LayerNetwork < ContextNetwork

  def createLayersWithContextLayerArchitecture
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, typeOfLink = args[:typeOfLink], args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer1Neurons])
    hiddenLayer1.each {|aNeuron| aNeuron.neuronControllingLearning = theBiasNeuron}  # 'placeholder' -- always on
    self.allNeuronLayers << hiddenLayer1

    hiddenLayer2 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer2Neurons])
    hiddenLayer2.each_with_index do |aNeuron, index|
      indexToNeuronControllingNeuronInPrecedingLayer = (index / 2).to_i
      aNeuron.neuronControllingLearning = hiddenLayer1[indexToNeuronControllingNeuronInPrecedingLayer]
      aNeuron.reverseLearningProbability = index.odd?
    end
    self.allNeuronLayers << hiddenLayer2

    self.outputLayer = createAndConnectLayer( (hiddenLayer1 + hiddenLayer2), typeOfNeuron = args[:typeOfOutputNeuron], typeOfLink = args[:typeOfLinkToOutput], args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end

end



class SelfOrg1NeuronNetwork < BaseNetwork

  attr_accessor :hiddenLayers, :hiddenNeurons

  def createNetwork
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, typeOfLink = args[:typeOfLink], args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer1Neurons])
    hiddenLayer1.each {|aNeuron| aNeuron.neuronControllingLearning = theBiasNeuron}  # 'placeholder' -- always on
    self.allNeuronLayers << hiddenLayer1

    connectAllLearningNeuronsToBiasNeuron
  end

  def connectAllLearningNeuronsToBiasNeuron
    connectAllNeuronsToBiasNeuronExceptForThe(inputLayer)
   end

  def addLinksFromBiasNeuronTo( neurons, typeOfLink)
    connect_layer_to_another([theBiasNeuron], neurons, typeOfLink, args)
  end



  #def createLayersWithContextLayerArchitecture
  #  self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, typeOfLink = args[:typeOfLink], args[:numberOfInputNeurons])
  #  self.allNeuronLayers << inputLayer
  #
  #  hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer1Neurons])
  #  hiddenLayer1.each {|aNeuron| aNeuron.neuronControllingLearning = theBiasNeuron}  # 'placeholder' -- always on
  #  self.allNeuronLayers << hiddenLayer1
  #end

end









######################################## Code Below is not being used.
#
#class JumpLinked4LayerNetwork < BaseNetwork
#  attr_accessor :hiddenLayer1, :hiddenLayer2
#
#  def createNetwork
#    createLayersAndSequentialLayerToLayerConnections
#    connect_layer_to_another(inputLayer, hiddenLayer2, args) # This is where we create links that 'jump across' hidden layer1.
#    connect_layer_to_another(hiddenLayer1, outputLayer, args) # This is where we create links that 'jump across' hidden layer2.
#    connectAllNeuronsToBiasNeuronExceptForThe(inputLayer)
#  end
#
#  def createLayersAndSequentialLayerToLayerConnections
#    self.inputLayer = createAndConnectLayer(nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
#    self.allNeuronLayers << inputLayer
#
#    self.hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], args[:numberOfHiddenLayer1Neurons])
#    self.allNeuronLayers << hiddenLayer1
#
#    self.hiddenLayer2 = createAndConnectLayer(hiddenLayer1, typeOfNeuron = args[:typeOfNeuron], args[:numberOfHiddenLayer2Neurons])
#    self.allNeuronLayers << hiddenLayer2
#
#    self.outputLayer = createAndConnectLayer(hiddenLayer2, typeOfNeuron = args[:typeOfOutputNeuron], args[:numberOfOutputNeurons])
#    self.allNeuronLayers << outputLayer
#  end
#end
#
#
#class SharedJumpLinked4LayerNetwork < JumpLinked4LayerNetwork
#
#  def createNetwork
#    createLayersAndSequentialLayerToLayerConnections
#    connect_layer_to_another(inputLayer, hiddenLayer2, args) # This is where we create links that 'jump across' the hidden layer1.
#    connect_layer_to_another(hiddenLayer1, outputLayer, args) # This is where we create links that 'jump across' the hidden layer2.
#    connectAllNeuronsToBiasNeuronExceptForThe(inputLayer)
#  end
#
#end
#
#
#
#class Recurrent2HiddenLayerNetworkSpecial < BaseNetwork
#  attr_accessor :hiddenLayer1, :hiddenLayer2, :linksBetweenHidden2Layers
#
#  def createStandardNetworkWithStandardFullyConnectedArchitecture
#    STDERR.puts "Error: number of neurons in hidden layers are not identical" if (args[:numberOfHiddenLayer1Neurons] != args[:numberOfHiddenLayer2Neurons])
#
#    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
#    self.allNeuronLayers << inputLayer
#
#    # self.hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingNeuronStepIO, args[:numberOfHiddenLayer1Neurons])
#    # self.hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingSymmetricalNeuron, args[:numberOfHiddenLayer1Neurons])
#    self.hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingNeuron, args[:numberOfHiddenLayer1Neurons])
#    self.allNeuronLayers << hiddenLayer1
#
#    # self.hiddenLayer2 = createAndConnectLayer((inputLayer + hiddenLayer1), typeOfNeuron = FlockingSymmetricalNeuron, args[:numberOfHiddenLayer2Neurons])
#    self.hiddenLayer2 = createAndConnectLayer((inputLayer + hiddenLayer1), typeOfNeuron = FlockingNeuron, args[:numberOfHiddenLayer2Neurons])
#    self.allNeuronLayers << hiddenLayer2
#
#    self.outputLayer = createAndConnectLayer(hiddenLayer2, typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
#    self.allNeuronLayers << outputLayer
#
#    connectAllNeuronsToBiasNeuronExceptForThe(inputLayer)
#  end
#
#  def modificationsToStandardNetworkArchitecture
#    # Weight Sharing code
#    inputLayerIncludingLinkFromBias = inputLayer + [theBiasNeuron]
#    shareWeightBetweenCorrespondingLinks(inputLayerIncludingLinkFromBias, hiddenLayer1,
#                                         inputLayerIncludingLinkFromBias, hiddenLayer2)
#
#    # to create just cross-connections between 2 hidden layers of a "simulated recurrent net" we need to delete ALL (direct recurrent: N1out to N1in connections)
#    # In other words, we delete the connection between a neuron's output and its input. i.e. we eliminate the "cat chases its tail" links.
#    deleteRecurrentSelfConnections(hiddenLayer1, hiddenLayer2)
#
#    # connectToAllNeuronsInReceivingLayer(theBiasNeuron, hiddenLayer2, args)
#
#    linksBetweenHidden2Layers = retrieveLinksBetweenGroupsOfNeurons(hiddenLayer1, hiddenLayer2)
#    linksBetweenHidden2Layers.each { |aLink| aLink.weight = 0.0 } ## Set inter-hidden-layer weights to zero
#  end
#end


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




