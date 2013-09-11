### VERSION "nCore"
## ../nCore/lib/core/NetworkFactories.rb

require_relative 'Utilities'

############################################################
class BaseNetwork
  attr_accessor :args, :allNeuronLayers, :theBiasNeuron,
                :inputLayer, :outputLayer,

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

class SimpleFlockingNeuronNetwork < BaseNetwork

  def createAllLayersOfNeurons
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    self.outputLayer = createAndConnectLayer(inputLayer, typeOfNeuron = FlockingOutputNeuron, args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end

############################################################

#@networkRecorder = NetworkRecorder.new(self, args
#
#def calcNetworksMeanSquareError
#  outputLayer = allNeuronLayers.last
#  sse = outputLayer.inject(0.0) { |sum, anOutputNeuron| sum + anOutputNeuron.calcSumOfSquaredErrors }
#  numberOfOutputNeurons = outputLayer.length
#  self.mse = (sse / (numberOfOutputNeurons * numberOfExamples))
#end
#
## Secondary Importance:
#
#def recordResponses
#  networkRecorder.recordResponses
#end
#
#def measures
#  networkRecorder.measures
#end
#
#

class NetworkRecorder
  attr_accessor :network, :args, :measures

  def initialize(network, args)
    @network = network
    @args = args
    @measures = []
    @trainingSequence = nil
  end

  def trainingSequence
    args[:trainingSequence]
  end

  def recordResponses
    measures << {:mse => network.mse, :epochs => trainingSequence.epochs} if (trainingSequence.timeToRecordData)
  end
end


