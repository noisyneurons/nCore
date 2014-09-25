### VERSION "nCore"
## ../nCore/lib/core/NeuralSelfOrg.rb

require_relative 'Utilities'
require_relative 'NeuralParts'


class TrainerSelfOrgWithLinkNormalization < TrainerBase

  def train
    distributeSetOfExamples(examples)
    totalEpochs = 0

    learningLayers = propagatingLayers = [ allNeuronLayers[1] ]
    attachLearningStrategy(propagatingLayers, Normalize)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    #attachLearningStrategy(propagatingLayers, SelfOrgLearning)
    #mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    forEachExampleDisplayInputsAndOutputs
    return totalEpochs, mse, calcTestingMeanSquaredErrors
  end


  def trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
    mse = 1e100
    while ((mse >= minMSE) && trainingSequence.stillMoreEpochs)
      propagateAndLearnForAnEpoch(propagatingLayers, learningLayers)
      trainingSequence.nextEpoch
      mse = calcMeanSumSquaredErrors
      currentEpochNumber = trainingSequence.epochs + totalEpochs
      puts "current epoch number= #{currentEpochNumber}\tmse = #{mse}" if (currentEpochNumber % 100 == 0)
    end
    totalEpochs += trainingSequence.epochs
    trainingSequence.startNextPhaseOfTraining
    return mse, totalEpochs
  end

  #allNeuronLayers[1].each { |aNeuron| aNeuron.afterSelfOrgReCalcLinkWeights }
  #resetAllNormalizationVariables(allNeuronLayers[1])

  def propagateAndLearnForAnEpoch(propagatingLayers, learningLayers)
    initializeStartOfEpochAcross(learningLayers)
    numberOfExamples.times do |exampleNumber|
      propagateExampleAcross(propagatingLayers, exampleNumber)
      learnExampleIn(learningLayers, exampleNumber)
    end
    endEpochAcross(learningLayers)
  end

  def initializeStartOfEpochAcross(layers)
    layers.each do |neurons|
      neurons.each { |neuron| neuron.startEpoch }
    end
  end

  def propagateExampleAcross(propagatingLayers, exampleNumber)
    propagatingLayers.each do |neurons|
      neurons.each { |aNeuron| aNeuron.propagate(exampleNumber) }
    end
  end

  def learnExampleIn(learningLayers, exampleNumber)
    learningLayers.reverse.each do |neurons|
      neurons.each { |aNeuron| aNeuron.learnExample }
    end
  end

  def endEpochAcross(layers)
    layers.each do |neurons|
      neurons.each { |neuron| neuron.endEpoch }
    end
  end

  def attachLearningStrategy(layers, learningStrategy)
    layers.each do |neurons|
      neurons.each { |neuron| neuron.learningStrat = learningStrategy.new(neuron) }
    end
  end

  ###########################

  #def performSelfOrgTrainingOn(layerOfNeurons)
  #  acrossExamplesAccumulateSelfOrgDeltaWs(layerOfNeurons)
  #  layerOfNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
  #end
  #
  #def acrossExamplesAccumulateSelfOrgDeltaWs(layerOfNeurons)
  #  startEpoch()
  #  numberOfExamples.times do |exampleNumber|
  #    propagateExampleUpToLayer(exampleNumber, layerOfNeurons)
  #    layerOfNeurons.each { |aNeuron| aNeuron.calcSelfOrgError }
  #
  #    layerOfNeurons.each do |aNeuron|
  #      dataRecord = aNeuron.recordResponsesForExample
  #      aNeuron.calcDeltaWsAndAccumulate
  #      aNeuron.dbStoreDetailedData
  #    end
  #  end
  #end
  #
  #def normalize(layer)
  #  resetAllNormalizationVariables(layer)
  #  numberOfExamples.times do |exampleNumber|
  #    propagateForNormalizationToLayer(exampleNumber, layer)
  #  end
  #  calculateNormalizationCoefficients(layer)
  #end
  #
  #def resetAllNormalizationVariables(layer)
  #  layer.each { |aNeuron| aNeuron.resetAllNormalizationVariables }
  #end
  #
  #def propagateForNormalizationToLayer(exampleNumber, lastLayerOfNeuronsToReceivePropagation)
  #  allNeuronLayers.each do |aLayer|
  #    if aLayer == lastLayerOfNeuronsToReceivePropagation
  #      aLayer.each { |aNeuron| aNeuron.propagateForNormalization(exampleNumber) }
  #    else
  #      aLayer.each { |aNeuron| aNeuron.propagate(exampleNumber) }
  #    end
  #    break if lastLayerOfNeuronsToReceivePropagation == aLayer
  #  end
  #end
  #
  #def calculateNormalizationCoefficients(layer)
  #  layer.each { |aNeuron| aNeuron.calculateNormalizationCoefficients }
  #end
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

module SelfOrganization # Module for Neuron classes

  def calcSelfOrgError
    targetPlus = 2.5
    targetMinus = -1.0 * targetPlus
    distanceBetweenTargets = targetPlus - targetMinus
    self.error = -1.0 * ioDerivativeFromNetInput(netInput) * (((netInput - targetMinus)/distanceBetweenTargets) - 0.5)
  end

  def resetAllNormalizationVariables
    inputLinks.each { |aLink| aLink.resetAllNormalizationVariables }
  end

  def propagateForNormalization(exampleNumber)
    self.exampleNumber = exampleNumber
    self.netInput = inputLinks.inject(0.0) { |sum, link| sum + link.propagateForNormalization }
    self.output = ioFunction(netInput)
  end

  def calculateNormalizationCoefficients
    inputLinks.each { |aLink| aLink.calculateNormalizationCoefficients }
  end

  def afterSelfOrgReCalcLinkWeights
    biasWeight = inputLinks.inject(0.0) { |sum, link| sum + link.propagateUsingZeroInput }
    inputLinks.each { |aLink| aLink.afterSelfOrgReCalcLinkWeights }
    inputLinks[-1].weight = biasWeight
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

  def afterSelfOrgReCalcLinkWeights
    puts "weightBefore= #{weight}"
    self.weight = normalizationMultiplier * weight
    puts "normalizationMultiplier= #{normalizationMultiplier}"
    puts "weightAfter= #{weight}"
  end

  def propagateUsingZeroInput
    return -1.0 * normalizationMultiplier * weight * normalizationOffset
  end
end







