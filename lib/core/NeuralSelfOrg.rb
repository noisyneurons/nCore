### VERSION "nCore"
## ../nCore/lib/core/NeuralSelfOrg.rb


class TrainerSelfOrgWithLinkNormalization < TrainerBase

  def train
    learningLayers = [allNeuronLayers[1]]
    propagatingLayers = allNeuronLayers
    attachLearningStrategy(learningLayers, Normalization)
    specifyIOFunction(learningLayers, NonMonotonicIOFunction)

    distributeSetOfExamples(examples)
    totalEpochs = 0
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    attachLearningStrategy(learningLayers, SelfOrgStrat)
    specifyIOFunction(learningLayers, NonMonotonicIOFunction)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
    calcWeightsForUNNormalizedInputs(learningLayers)

    forEachExampleDisplayInputsAndOutputs

    return totalEpochs, mse, 0.998 # calcTestingMeanSquaredErrors
  end

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer])
  end

  def trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
    mse = 1e100
    while ((mse >= minMSE) && trainingSequence.stillMoreEpochs)
      propagateAndLearnForAnEpoch(propagatingLayers, learningLayers)
      trainingSequence.nextEpoch
      mse = 0.997 # = calcMeanSumSquaredErrors
      currentEpochNumber = trainingSequence.epochs + totalEpochs
      puts "current epoch number= #{currentEpochNumber}\tmse = #{mse}" if (currentEpochNumber % 100 == 0)
    end
    totalEpochs += trainingSequence.epochs
    trainingSequence.startNextPhaseOfTraining
    return mse, totalEpochs
  end

  def calcWeightsForUNNormalizedInputs(learningLayers)
    learningLayers.each { |neurons| neurons.each { |aNeuron| aNeuron.calcWeightsForUNNormalizedInputs } }
  end
end


class Trainer1SelfOrgAndContext < TrainerSelfOrgWithLinkNormalization
  def train
    learningLayers = [allNeuronLayers[1]]
    propagatingLayers = allNeuronLayers

    normalizationStrategy = Normalization.new()
    contextAdapter = AdapterForContext.new(strategy, theEnclosingNeuron, contextController)
    attachLearningStrategy(learningLayers, Normalization)
    specifyIOFunction(learningLayers, NonMonotonicIOFunction)

    distributeSetOfExamples(examples)
    totalEpochs = 0
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    attachLearningStrategy(learningLayers, SelfOrgStrat)
    specifyIOFunction(learningLayers, NonMonotonicIOFunction)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
    calcWeightsForUNNormalizedInputs(learningLayers)

    forEachExampleDisplayInputsAndOutputs

    return totalEpochs, mse, 0.998 # calcTestingMeanSquaredErrors
  end


end

########################################################################
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

########################################################################
########################################################################

class LinkWithNormalization < Link
  attr_accessor :inputsOverEpoch, :normalizationOffset, :largestAbsoluteArrayElement, :normalizationMultiplier

  def initialize(inputNeuron, outputNeuron, args)
    super(inputNeuron, outputNeuron, args)
    @inputsOverEpoch = []
    resetAllNormalizationVariables
  end

  def resetAllNormalizationVariables
    self.inputsOverEpoch.clear
    self.normalizationOffset = 0.0
    self.largestAbsoluteArrayElement = 1.0
    self.normalizationMultiplier = 1.0
  end

  def storeEpochHistory
    self.inputsOverEpoch << inputNeuron.output
  end

  def propagate
    return normalizationMultiplier * weight * (inputNeuron.output - normalizationOffset)
  end

  def calculateNormalizationCoefficients
    averageOfInputs = inputsOverEpoch.mean
    self.normalizationOffset = averageOfInputs
    centeredArray = inputsOverEpoch.collect { |value| value - normalizationOffset }
    largestAbsoluteArrayElement = centeredArray.minmax.abs.max.to_f
    self.normalizationMultiplier = if largestAbsoluteArrayElement > 1.0e-5
                                     1.0 / largestAbsoluteArrayElement
                                   else
                                     0.0
                                   end
  end

  def calcWeightsForUNNormalizedInputs
    puts "weightBefore= #{weight}"
    self.weight = normalizationMultiplier * weight
    puts "normalizationMultiplier= #{normalizationMultiplier}"
    puts "weightAfter= #{weight}"
  end

  def propagateUsingZeroInput
    return -1.0 * normalizationMultiplier * weight * normalizationOffset
  end

  def to_s
    return "Weight=\t#{weight}\tOffset=\t#{normalizationOffset}\tMultiplier=\t#{normalizationMultiplier}\tDeltaW=\t#{deltaW}\tAccumulatedDeltaW=\t#{deltaWAccumulated}\tWeightAtBeginningOfTraining=\t#{weightAtBeginningOfTraining}\tFROM: #{inputNeuron.class.to_s} #{inputNeuron.id} TO: #{outputNeuron.class.to_s} #{outputNeuron.id}"
  end
end







