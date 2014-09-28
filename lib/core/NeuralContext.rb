### VERSION "nCore"
## ../nCore/lib/core/NeuralContext.rb

########################################################################

class Trainer1SelfOrgAndContext < TrainerSelfOrgWithLinkNormalization
  def train
    distributeSetOfExamples(examples)
    totalEpochs = 0

    learningLayers = [allNeuronLayers[1]]
    propagatingLayers = allNeuronLayers

    anInputNeuron = allNeuronLayers[0][0]
    learningController = DummyLearningController.new(anInputNeuron)

    strategyArguments = {:strategy => Normalization, :ioFunction => NonMonotonicIOFunction, :contextController => learningController}
    attachLearningStrategy(learningLayers, AdapterForContext, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    strategyArguments = {:strategy => SelfOrgStrat, :ioFunction => NonMonotonicIOFunction, :contextController => learningController}
    attachLearningStrategy(learningLayers, AdapterForContext, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    calcWeightsForUNNormalizedInputs(learningLayers)

    forEachExampleDisplayInputsAndOutputs

    return totalEpochs, mse, 0.998 # calcTestingMeanSquaredErrors
  end
end


########################################################################

class Trainer2SelfOrgAndContext < TrainerSelfOrgWithLinkNormalization
  def train
    distributeSetOfExamples(examples)
    totalEpochs = 0

    inputLayer = allNeuronLayers[0]
    hiddenLayer1 = allNeuronLayers[1]
    hiddenLayer2 = allNeuronLayers[2]


    ### Now will self-org 1st hidden layer

    learningLayers = [hiddenLayer1]
    propagatingLayers = [inputLayer, hiddenLayer1]

    strategyArguments = {:ioFunction => NonMonotonicIOFunction}

    attachLearningStrategy(learningLayers, Normalization, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    attachLearningStrategy(learningLayers, SelfOrgStrat, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    ### Now will self-org 2nd hidden layer

    learningLayers = [hiddenLayer2]
    propagatingLayers = [inputLayer, hiddenLayer1, hiddenLayer2]

    theOnlyNeuronInHiddenLayer1 = hiddenLayer1[0]
    learningController = LearningControlledByNeuron.new(theOnlyNeuronInHiddenLayer1)

    strategyArguments = {:strategy => Normalization, :ioFunction => NonMonotonicIOFunction, :contextController => learningController}
    attachLearningStrategy(learningLayers, AdapterForContext, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    strategyArguments = {:strategy => SelfOrgStrat, :ioFunction => NonMonotonicIOFunction, :contextController => learningController}
    attachLearningStrategy(learningLayers, AdapterForContext, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)





    layersThatWereNormalized = [hiddenLayer1, hiddenLayer2]
    calcWeightsForUNNormalizedInputs(layersThatWereNormalized)

    forEachExampleDisplayInputsAndOutputs

    return totalEpochs, mse, 0.998 # calcTestingMeanSquaredErrors
  end
end


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
    self.allNeuronLayers << hiddenLayer1

    hiddenLayer2 = createAndConnectLayer(inputLayer, typeOfNeuron = args[:typeOfNeuron], typeOfLink = args[:typeOfLink], args[:numberOfHiddenLayer2Neurons])
    #hiddenLayer2.each_with_index do |aNeuron, index|
    #  indexToNeuronControllingNeuronInPrecedingLayer = (index / 2).to_i
    #  controllingNeuron = hiddenLayer1[indexToNeuronControllingNeuronInPrecedingLayer]
    #  aNeuron.learningController = if index.even?
    #                                 LearningControlledByNeuron.new(controllingNeuron)
    #                               else
    #                                 LearningControlledByNeuronOutputReversed.new(controllingNeuron)
    #                               end
    #end
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
    #  aGroupOfExampleNumbers =  [0,1,2,3,8,9,10,11]  # (0..7).to_a #
    aGroupOfExampleNumbers = (0..15).to_a
    if aGroupOfExampleNumbers.include?(sensor.exampleNumber)
      1.0
    else
      0.0
    end
  end
end




