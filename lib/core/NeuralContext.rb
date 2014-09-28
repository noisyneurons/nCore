### VERSION "nCore"
## ../nCore/lib/core/NeuralContext.rb

require_relative 'Utilities'
require_relative 'NeuralParts'


###########################################################################################

class ContextNetwork < BaseNetwork
  attr_accessor :hiddenLayers, :hiddenNeurons

  def createNetwork
    createLayersWithContextLayerArchitecture
    self.hiddenLayers = allNeuronLayers[1..-2]
    self.hiddenNeurons = hiddenLayers.flatten
    connectAllLearningNeuronsToBiasNeuron
  end

  def connectAllLearningNeuronsToBiasNeuron
    addLinksFromBiasNeuronTo(hiddenNeurons, args[:typeOfLink])
    addLinksFromBiasNeuronTo(outputLayer, args[:typeOfLinkToOutput])
  end

  def addLinksFromBiasNeuronTo(neurons, typeOfLink)
    connect_layer_to_another([theBiasNeuron], neurons, typeOfLink, args)
  end
end


class Context4LayerNetwork < ContextNetwork

  def createLayersWithContextLayerArchitecture
    self.inputLayer = createAndConnectLayer(inputLayerToLayerToBeCreated = nil, typeOfNeuron= InputNeuron, typeOfLink = args[:typeOfLink], args[:numberOfInputNeurons])
    self.allNeuronLayers << inputLayer

    hiddenLayer1 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer1Neurons])
    hiddenLayer1.each { |aNeuron| aNeuron.learningController = LearningController.new }
    self.allNeuronLayers << hiddenLayer1

    hiddenLayer2 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer2Neurons])
    hiddenLayer2.each_with_index do |aNeuron, index|
      indexToNeuronControllingNeuronInPrecedingLayer = (index / 2).to_i
      controllingNeuron = hiddenLayer1[indexToNeuronControllingNeuronInPrecedingLayer]
      aNeuron.learningController = if index.even?
                                     LearningControlledByNeuron.new(controllingNeuron)
                                   else
                                     LearningControlledByNeuronOutputReversed.new(controllingNeuron)
                                   end
    end
    self.allNeuronLayers << hiddenLayer2

    self.outputLayer = createAndConnectLayer((hiddenLayer1 + hiddenLayer2), typeOfNeuron = args[:typeOfOutputNeuron], typeOfLink = args[:typeOfLinkToOutput], args[:numberOfOutputNeurons])
    self.allNeuronLayers << outputLayer
  end
end



class LearningController
  attr_accessor :sensor

  def initialize(sensor=nil)
    @sensor = sensor
  end

  def output
    1.0
  end
end


class LearningControlledByNeuron < LearningController
  def output
    transform(sensor.output)
  end

  def transform(input)
    if input >= 0.5
      1.0
    else
      0.0
    end
  end
end

class LearningControlledByNeuronOutputReversed < LearningControlledByNeuron
  def transform(input)
    1.0 - super(input)
  end
end

class DummyLearningController < LearningController
  def output
    aGroupOfExampleNumbers =  [0,1,2,3,8,9,10,11]  # (0..7).to_a #
    if aGroupOfExampleNumbers.include?(sensor.exampleNumber)
      1.0
    else
      0.0
    end
  end
end




