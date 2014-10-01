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
    controllingLayers = nil
    propagatingLayers = [inputLayer, hiddenLayer1]
    ioFunction = NonMonotonicIOFunction
    totalEpochs, mse = normalizationAndSelfOrgTraining(learningLayers, controllingLayers, propagatingLayers, ioFunction, totalEpochs)


    learningLayers = [hiddenLayer2]
    controllingLayers = [hiddenLayer1]
    propagatingLayers = [inputLayer, hiddenLayer1, hiddenLayer2]
    ioFunction = NonMonotonicIOFunction
    totalEpochs, mse = normalizationAndSelfOrgTraining(learningLayers, controllingLayers, propagatingLayers, ioFunction, totalEpochs)

    #forEachExampleDisplayInputsAndOutputs
    return totalEpochs, mse, 0.998 # calcTestingMeanSquaredErrors
  end


  def normalizationAndSelfOrgTraining(learningLayers, controllingLayers, propagatingLayers, ioFunction, totalEpochs)

    strategyArguments = {}
    strategyArguments[:ioFunction] = ioFunction

    if (controllingLayers.nil?)
      attachLearningStrategy(learningLayers, Normalization, strategyArguments)
      mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

      attachLearningStrategy(learningLayers, SelfOrgStrat, strategyArguments)
      mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
    else

      ### Now will self-org 2nd hidden layer

      theOnlyNeuronInHiddenLayer1 = controllingLayers[0][0]

      ## Normalization of both 2nd hidden layer neurons
      strategyArguments[:strategy] = Normalization
      learningController = LearningControlledByNeuron.new(theOnlyNeuronInHiddenLayer1)
      strategyArguments[:contextController] = learningController
      firstNeuronInHiddenLayer2 = learningLayers[0][0]
      attachLearningStrategy([[firstNeuronInHiddenLayer2]], AdapterForContext, strategyArguments)

      learningControllerNOT = LearningControlledByNeuronNOT.new(theOnlyNeuronInHiddenLayer1)
      strategyArguments[:contextController] = learningControllerNOT
      secondNeuronInHiddenLayer2 = learningLayers[0][1]
      attachLearningStrategy([[secondNeuronInHiddenLayer2]], AdapterForContext, strategyArguments)

      mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
      #### end normalization

      ## Self-Org of both 2nd hidden layer neurons
      strategyArguments[:strategy] = SelfOrgStrat
      strategyArguments[:contextController] = learningController
      attachLearningStrategy([[firstNeuronInHiddenLayer2]], AdapterForContext, strategyArguments)

      strategyArguments[:contextController] = learningControllerNOT
      attachLearningStrategy([[secondNeuronInHiddenLayer2]], AdapterForContext, strategyArguments)

      mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
      #### end self-organization
    end

    return totalEpochs, mse
  end


end


########################################################################

class Trainer3SelfOrgAndContext < Trainer2SelfOrgAndContext
  def train
    distributeSetOfExamples(examples)
    inputLayer = allNeuronLayers[0]
    hiddenLayer1 = allNeuronLayers[1]
    hiddenLayer2 = allNeuronLayers[2]

    totalEpochs = 0

    ### Now will self-org 1st hidden layer
    learningLayers = [hiddenLayer1]
    controllingLayers = nil
    propagatingLayers = [inputLayer, hiddenLayer1]
    ioFunction = NonMonotonicIOFunction
    totalEpochs, mse = normalizationAndSelfOrgTraining(learningLayers, controllingLayers, propagatingLayers, ioFunction, totalEpochs)


    learningLayers = [hiddenLayer2]
    controllingLayers = [hiddenLayer1]
    propagatingLayers = [inputLayer, hiddenLayer1, hiddenLayer2]
    ioFunction = NonMonotonicIOFunction
    totalEpochs, mse = normalizationAndSelfOrgTraining(learningLayers, controllingLayers, propagatingLayers, ioFunction, totalEpochs)


    layersThatWereNormalized = [hiddenLayer1, hiddenLayer2]
    calcWeightsForUNNormalizedInputs(layersThatWereNormalized)

    totalEpochs, mse = supervisedTraining(totalEpochs)

    forEachExampleDisplayInputsAndOutputs(outputLayer)

    return totalEpochs, mse, calcTestingMeanSquaredErrors
  end

  def supervisedTraining(totalEpochs)

    inputLayer = allNeuronLayers[0]
    hiddenLayer1 = allNeuronLayers[1]
    hiddenLayer2 = allNeuronLayers[2]
    outputLayer = allNeuronLayers[3]

    ### Now will self-org 1st hidden layer

    learningLayers = [outputLayer]
    propagatingLayers = [inputLayer, hiddenLayer1, hiddenLayer2, outputLayer]

    strategyArguments = {}
    strategyArguments[:ioFunction] = NonMonotonicIOFunction

    attachLearningStrategy(learningLayers, LearningBPOutput, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    return totalEpochs, mse
  end

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
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
    #  aLearningController = LearningControlledByNeuron.new(controllingNeuron)
    #  aNeuron.learningController = if index.even?
    #                                 aLearningController
    #                               else
    #                                 AdapterForLearningController.new(aLearningController)
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

class LearningControlledByNeuronNOT < LearningControlledByNeuron
  def output
    logicalNOT(transform(sensor.output))
  end

  protected

  def logicalNOT(input)
    returnValue = if input == 1.0
                    0.0
                  else
                    1.0
                  end
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




