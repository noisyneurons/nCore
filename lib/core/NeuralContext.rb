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

  attr_accessor :hiddenLayer1, :hiddenLayer2

  def postInitialize
    @hiddenLayer1 = allNeuronLayers[1]
    @hiddenLayer2 = allNeuronLayers[2]
  end

  def train
    distributeSetOfExamples(examples)

    totalEpochs = 0

    ### Now will self-org 1st hidden layer
    learningLayers = [hiddenLayer1]
    controllingLayers = nil
    propagatingLayers = [inputLayer, hiddenLayer1]
    ioFunction = NonMonotonicIOFunction
    totalEpochs, mse = normalizationAndSelfOrgTraining(learningLayers, controllingLayers, propagatingLayers, ioFunction, totalEpochs)


    ### Now will self-org 2nd hidden layer
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
      mse, totalEpochs = normalizationAndSelfOrgWITHOUTContext(learningLayers, propagatingLayers, strategyArguments, totalEpochs)
    else
      mse, totalEpochs = normalizationAndSelfOrgWithContext(learningLayers, controllingLayers, propagatingLayers, strategyArguments, totalEpochs)
    end
    return totalEpochs, mse
  end


  def normalizationAndSelfOrgWithContext(learningLayers, controllingLayers, propagatingLayers, strategyArguments, totalEpochs)

    singleLayerControllingLearning = controllingLayers[0]
    singleLearningLayer = learningLayers[0]

    #### attaching normalization strategy WITH CONTEXT ADAPTER
    singleLayerControllingLearning.each_with_index do |neuronInControllingLayer, indexToControlNeuron|
      indexToLearningNeuron = 2 * indexToControlNeuron

      strategyArguments[:strategy] = Normalization
      learningController = LearningControlledByNeuron.new(neuronInControllingLayer)
      strategyArguments[:contextController] = learningController
      aLearningNeuron = singleLearningLayer[indexToLearningNeuron]
      attachLearningStrategy([[aLearningNeuron]], AdapterForContext, strategyArguments)

      learningControllerNOT = LearningControlledByNeuronNOT.new(neuronInControllingLayer)
      strategyArguments[:contextController] = learningControllerNOT
      aLearningNeuron = singleLearningLayer[indexToLearningNeuron + 1]
      attachLearningStrategy([[aLearningNeuron]], AdapterForContext, strategyArguments)
    end

    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
    #### end normalization WITH CONTEXT


    #### attaching Self-Org strategy WITH CONTEXT ADAPTER
    singleLayerControllingLearning.each_with_index do |neuronInControllingLayer, indexToControlNeuron|
      indexToLearningNeuron = 2 * indexToControlNeuron

      strategyArguments[:strategy] = SelfOrgStrat
      learningController = LearningControlledByNeuron.new(neuronInControllingLayer)
      strategyArguments[:contextController] = learningController
      aLearningNeuron = singleLearningLayer[indexToLearningNeuron]
      attachLearningStrategy([[aLearningNeuron]], AdapterForContext, strategyArguments)

      learningControllerNOT = LearningControlledByNeuronNOT.new(neuronInControllingLayer)
      strategyArguments[:contextController] = learningControllerNOT
      aLearningNeuron = singleLearningLayer[indexToLearningNeuron + 1]
      attachLearningStrategy([[aLearningNeuron]], AdapterForContext, strategyArguments)
    end

    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
    return mse, totalEpochs
  end


  def normalizationAndSelfOrgWITHOUTContext(learningLayers, propagatingLayers, strategyArguments, totalEpochs)
    attachLearningStrategy(learningLayers, Normalization, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    attachLearningStrategy(learningLayers, SelfOrgStrat, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
    return mse, totalEpochs
  end
end


########################################################################

class Trainer3SelfOrgAndContext < Trainer2SelfOrgAndContext
  def train
    distributeSetOfExamples(examples)

    totalEpochs = 0

    ### Now will self-org 1st hidden layer
    learningLayers = [hiddenLayer1]
    controllingLayers = nil
    propagatingLayers = [inputLayer, hiddenLayer1]
    ioFunction = NonMonotonicIOFunction
    totalEpochs, mse = normalizationAndSelfOrgTraining(learningLayers, controllingLayers, propagatingLayers, ioFunction, totalEpochs)


    ### Now will self-org 2nd hidden layer
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




