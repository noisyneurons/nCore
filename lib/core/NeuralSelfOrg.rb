### VERSION "nCore"
## ../nCore/lib/core/NeuralSelfOrg.rb

require_relative 'Utilities'
require_relative 'NeuralParts'


class TrainerSelfOrgWithLinkNormalization < TrainerBase

  def train
    distributeSetOfExamples(examples)

    puts "phase1: self-org for hidden layer 1 "
    puts "allNeuronLayers[1][0].output= #{allNeuronLayers[1][0].output}"

    normalize(allNeuronLayers[1])

    [0, 1, 2].each do |i|
      puts "allNeuronLayers[1][0].inputLinks[#{i}].weight= #{allNeuronLayers[1][0].inputLinks[i].weight}"
      puts "allNeuronLayers[1][0].inputLinks[#{i}].normalizationOffset= #{allNeuronLayers[1][0].inputLinks[i].normalizationOffset}"
      puts "allNeuronLayers[1][0].inputLinks[#{i}].normalizationMultiplier= #{allNeuronLayers[1][0].inputLinks[i].normalizationMultiplier}"
    end

    phaseTrain { performSelfOrgTrainingOn(allNeuronLayers[1]) }
    forEachExampleDisplayInputsAndOutputs

    return trainingSequence.epochs, 9999.9, 9999.9
  end

  def phaseTrain
    mse = 1e100
    while ((mse >= minMSE) && trainingSequence.stillMoreEpochs)
      yield
      neuronsWithInputLinks.each { |aNeuron| aNeuron.dbStoreNeuronData }
      trainingSequence.nextEpoch
    end
    allNeuronLayers[1].each { |aNeuron| aNeuron.afterSelfOrgReCalcLinkWeights }
    resetAllNormalizationVariables(allNeuronLayers[1])
    trainingSequence.startNextPhaseOfTraining
    return mse
  end

  def performSelfOrgTrainingOn(layerOfNeurons)
    acrossExamplesAccumulateSelfOrgDeltaWs(layerOfNeurons)
    layerOfNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
  end

  def acrossExamplesAccumulateSelfOrgDeltaWs(layerOfNeurons)
    clearEpochAccumulationsInAllNeurons()
    numberOfExamples.times do |exampleNumber|
      propagateExampleUpToLayer(exampleNumber, layerOfNeurons)
      layerOfNeurons.each { |aNeuron| aNeuron.calcSelfOrgError }

      layerOfNeurons.each do |aNeuron|
        dataRecord = aNeuron.recordResponsesForExample
        aNeuron.calcDeltaWsAndAccumulate
        aNeuron.dbStoreDetailedData
      end
    end
  end

  def normalize(layer)
    resetAllNormalizationVariables(layer)
    numberOfExamples.times do |exampleNumber|
      propagateForNormalizationToLayer(exampleNumber, layer)
    end
    calculateNormalizationCoefficients(layer)
  end

  def resetAllNormalizationVariables(layer)
    layer.each { |aNeuron| aNeuron.resetAllNormalizationVariables }
  end

  def propagateForNormalizationToLayer(exampleNumber, lastLayerOfNeuronsToReceivePropagation)
    allNeuronLayers.each do |aLayer|
      if aLayer == lastLayerOfNeuronsToReceivePropagation
        aLayer.each { |aNeuron| aNeuron.propagateForNormalization(exampleNumber) }
      else
        aLayer.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      end
      break if lastLayerOfNeuronsToReceivePropagation == aLayer
    end
  end

  def calculateNormalizationCoefficients(layer)
    layer.each { |aNeuron| aNeuron.calculateNormalizationCoefficients }
  end

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer])
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
#    hiddenLayer1.each {|aNeuron| aNeuron.neuronControllingLearning = theBiasNeuron}  # 'placeholder' -- always on
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

module SelfOrganization

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
    # inputLinks[-1].setBiasLinkNormalizationCoefficients
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

  def propagateForNormalization
    inputForThisExample = inputNeuron.output
    self.inputsOverEpoch << inputForThisExample
    return inputForThisExample * weight
  end

  def propagate
    return normalizationMultiplier * weight * (inputNeuron.output - normalizationOffset)
  end

  def calculateNormalizationCoefficients
    puts "inputsOverEpoch= #{inputsOverEpoch}"
    averageOfInputs = inputsOverEpoch.mean
    puts "averageOfInputs= #{averageOfInputs}"

    self.normalizationOffset = averageOfInputs
    centeredArray = inputsOverEpoch.collect { |value| value - normalizationOffset }
    largestAbsoluteArrayElement = centeredArray.minmax.abs.max.to_f
    self.normalizationMultiplier = if largestAbsoluteArrayElement > 1.0e-5
                                     1.0 / largestAbsoluteArrayElement
                                   else
                                     0.0
                                   end
  end

  #def setBiasLinkNormalizationCoefficients
  #  self.normalizationMultiplier = 0.0
  #end

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







