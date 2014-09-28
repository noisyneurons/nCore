### VERSION "nCore"
## ../nCore/lib/core/Trainers.rb


###################################################################
###################################################################

class TrainingSequence
  attr_accessor :args, :epochs, :maxNumberOfEpochs,
                :stillMoreEpochs

  def initialize(args)
    @args = args
    @maxNumberOfEpochs = args[:maxNumEpochs]
    @epochs = -1
    @stillMoreEpochs = true
    nextEpoch
  end

  def epochs=(value)
    @epochs = value
    args[:epochs] = value
  end

  def nextEpoch
    self.epochs += 1
    self.stillMoreEpochs = areThereStillMoreEpochs?
  end

  def startNextPhaseOfTraining
    self.epochs = -1
    self.stillMoreEpochs = true
    nextEpoch
  end

  protected

  def areThereStillMoreEpochs?
    if epochs >= maxNumberOfEpochs
      false
    else
      true
    end
  end
end

class MultiPhaseTrainingSequence < TrainingSequence
  attr_accessor :maxEpochNumbersForEachPhase, :phaseIndex, :numberOfPhases

  def initialize(args)
    @args = args
    @maxEpochNumbersForEachPhase = args[:maxEpochNumbersForEachPhase]
    @numberOfPhases = @maxEpochNumbersForEachPhase.length
    @phaseIndex = -1
    startNextPhaseOfTraining
  end

  def startNextPhaseOfTraining
    self.phaseIndex += 1
    return if (numberOfPhases == phaseIndex)
    self.maxNumberOfEpochs = maxEpochNumbersForEachPhase[phaseIndex]
    self.epochs = -1
    self.stillMoreEpochs = true
    nextEpoch
  end
end


###################################################################
###################################################################

class TrainerBase
  attr_accessor :examples, :network, :numberOfOutputNeurons, :allNeuronLayers, :allNeuronsInOneArray, :inputLayer,
                :outputLayer, :theBiasNeuron, :neuronsWithInputLinks, :neuronsWithInputLinksInReverseOrder,
                :args, :trainingSequence, :numberOfExamples, :startTime, :elapsedTime, :minMSE
  include NeuronToNeuronConnection
  include ExampleDistribution

  def initialize(examples, network, args)
    @args = args

    @network = network
    @allNeuronLayers = network.allNeuronLayers
    @allNeuronsInOneArray = @allNeuronLayers.flatten
    @inputLayer = @allNeuronLayers[0]
    @outputLayer = @allNeuronLayers[-1]
    @theBiasNeuron = network.theBiasNeuron
    @neuronsWithInputLinks = (@allNeuronsInOneArray - @inputLayer).flatten
    @neuronsWithInputLinksInReverseOrder = @neuronsWithInputLinks.reverse
    @numberOfOutputNeurons = @outputLayer.length

    @examples = examples
    @numberOfExamples = examples.length

    @minMSE = args[:minMSE]
    @trainingSequence = args[:trainingSequence]

    @startTime = Time.now
    @elapsedTime = nil

    postInitialize
  end

  def postInitialize
  end

  def train
    learningLayers = allNeuronLayers - [inputLayer]
    propagatingLayers = allNeuronLayers

    strategyArgs = {:ioFunction => SigmoidIOFunction}
    attachLearningStrategy(learningLayers - [outputLayer], LearningBP, strategyArgs)
    attachLearningStrategy([outputLayer], LearningBPOutput, strategyArgs)

    distributeSetOfExamples(examples)
    totalEpochs = 0
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    forEachExampleDisplayInputsAndOutputs
    return totalEpochs, mse, calcTestingMeanSquaredErrors
  end

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
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

  def attachLearningStrategy(layers, learningStrategy, strategyArgs)
    layers.each do |neurons|
      neurons.each { |neuron| neuron.learningStrat = learningStrategy.new(neuron, strategyArgs) }
    end
  end


  ###########################

  ###------------  Core Metric Support Section ------------------------------

  def calcMeanSumSquaredErrors # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder
    return genericCalcMeanSumSquaredErrors(numberOfExamples)
  end

  def calcTestingMeanSquaredErrors
    testMSE = nil
    testingExamples = args[:testingExamples]
    numberOfTestingExamples = args[:numberOfTestingExamples]
    unless (testingExamples.nil?)
      distributeSetOfExamples(testingExamples)
      testMSE = genericCalcMeanSumSquaredErrors(numberOfTestingExamples)
      distributeSetOfExamples(examples)
    end
    return testMSE
  end

  def genericCalcMeanSumSquaredErrors(numberOfExamples)
    squaredErrors = []
    numberOfExamples.times do |exampleNumber|
      propagateAcrossEntireNetwork(exampleNumber)
      squaredErrors << calcWeightedErrorMetricForExample()
    end
    sse = squaredErrors.flatten.reduce(:+)
    return (sse / (numberOfExamples * numberOfOutputNeurons))
  end

  def forEachExampleDisplayInputsAndOutputs
    puts "At end of Training:"
    examples.each_with_index do |anExample, exampleNumber|
      inputs = anExample[:inputs]
      propagateAcrossEntireNetwork(exampleNumber)
      outputs = outputLayer.collect { |anOutputNeuron| anOutputNeuron.output }
      puts "\t\t\tinputs= #{inputs}\toutputs= #{outputs}"
    end
  end

  def propagateAcrossEntireNetwork(exampleNumber)
    allNeuronsInOneArray.flatten.each { |aNeuron| aNeuron.propagate(exampleNumber) }
  end

  def calcWeightedErrorMetricForExample
    outputLayer.collect { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
  end
end


###################################################################


#class TrainerProj2SelfOrgAndContext < TrainerBase
##  include SelfOrgTraining
#
#  def train
#    distributeSetOfExamples(examples)
#
#    puts "phase1: self-org for hidden layer 1 "
#    puts "allNeuronLayers[1][0].output= #{allNeuronLayers[1][0].output}"
#
#    normalize(allNeuronLayers[1])
#
#    [0, 1, 2].each do |i|
#      puts "allNeuronLayers[1][0].inputLinks[#{i}].weight= #{allNeuronLayers[1][0].inputLinks[i].weight}"
#      puts "allNeuronLayers[1][0].inputLinks[#{i}].normalizationOffset= #{allNeuronLayers[1][0].inputLinks[i].normalizationOffset}"
#      puts "allNeuronLayers[1][0].inputLinks[#{i}].normalizationMultiplier= #{allNeuronLayers[1][0].inputLinks[i].normalizationMultiplier}"
#    end
#
#    phaseTrain { performSelfOrgTrainingOn(allNeuronLayers[1]) }
#    forEachExampleDisplayInputsAndOutputs
#
#
#    forEachExampleDisplayInputsAndOutputs
#    testMSE = calcTestingMeanSquaredErrors
#    return trainingSequence.epochs, calcMSE, testMSE
#  end
#
#  def phaseTrain
#    mse = 1e100
#    while ((mse >= minMSE) && trainingSequence.stillMoreEpochs)
#      yield
#      neuronsWithInputLinks.each { |aNeuron| aNeuron.dbStoreNeuronData }
#      trainingSequence.nextEpoch
#    end
#    allNeuronLayers[1].each { |aNeuron| aNeuron.afterSelfOrgReCalcLinkWeights }
#    resetAllNormalizationVariables(allNeuronLayers[1])
#    trainingSequence.startNextPhaseOfTraining
#    return mse
#  end
#
#  def performSelfOrgTrainingOn(layerOfNeurons)
#    acrossExamplesAccumulateSelfOrgDeltaWs(layerOfNeurons)
#    layerOfNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
#  end
#
#  def acrossExamplesAccumulateSelfOrgDeltaWs(layerOfNeurons)
#    startEpoch()
#    numberOfExamples.times do |exampleNumber|
#      propagateExampleUpToLayer(exampleNumber, layerOfNeurons)
#      layerOfNeurons.each { |aNeuron| aNeuron.calcSelfOrgError }
#
#      layerOfNeurons.each do |aNeuron|
#        dataRecord = aNeuron.recordResponsesForExample
#        aNeuron.calcDeltaWsAndAccumulate
#        aNeuron.dbStoreDetailedData
#      end
#    end
#  end
#
#  def normalize(layer)
#    resetAllNormalizationVariables(layer)
#    numberOfExamples.times do |exampleNumber|
#      propagateForNormalizationToLayer(exampleNumber, layer)
#    end
#    calculateNormalizationCoefficients(layer)
#  end
#
#  def resetAllNormalizationVariables(layer)
#    layer.each { |aNeuron| aNeuron.resetAllNormalizationVariables }
#  end
#
#  def propagateForNormalizationToLayer(exampleNumber, lastLayerOfNeuronsToReceivePropagation)
#    allNeuronLayers.each do |aLayer|
#      if aLayer == lastLayerOfNeuronsToReceivePropagation
#        aLayer.each { |aNeuron| aNeuron.propagateForNormalization(exampleNumber) }
#      else
#        aLayer.each { |aNeuron| aNeuron.propagate(exampleNumber) }
#      end
#      break if lastLayerOfNeuronsToReceivePropagation == aLayer
#    end
#  end
#
#  def calculateNormalizationCoefficients(layer)
#    layer.each { |aNeuron| aNeuron.calculateNormalizationCoefficients }
#  end
#
#
#  def trainOLD
#    distributeSetOfExamples(examples)
#
#    #puts "phase1: short bp for hidden layer 1 + output layer"
#    learningNeurons = allNeuronLayers[1] + outputLayer
#    phaseTrain { performStandardBackPropTrainingOn(learningNeurons) }
#
#    puts "phase2: self-org for hidden layer 1 "
#    phaseTrain { performSelfOrgTrainingOn(allNeuronLayers[1]) }
#
#    #puts "phase3: short bp for hidden layer 1 + output layer"
#    #phaseTrain { performStandardBackPropTrainingOn(learningNeurons) }
#
#    #puts "phase4: context learning for hidden layer 2, controlled by hidden layer 1 "
#    #phaseTrain { performStandardBackPropTrainingOn(neuronsWithInputLinks) }
#
#    #puts "phase5: context learning for hidden layer 2, controlled by hidden layer 1"
#    #phaseTrain { performSelfOrgTrainingOn( allNeuronLayers[2] ) }
#    #
#    #puts "phase6"
#    #phaseTrain { performStandardBackPropTrainingOn(neuronsWithInputLinks) }
#    #
#    #allNeuronLayers[2].each do |aNeuron|
#    #  aNeuron.neuronControllingLearning = theBiasNeuron
#    #  aNeuron.reverseLearningProbability = false
#    #end
#    #
#    #puts "phase7"
#    #phaseTrain { performSelfOrgTrainingOn( allNeuronLayers[2] ) }
#    #
#    #puts "phase8"
#    #phaseTrain { performStandardBackPropTrainingOn(neuronsWithInputLinks) }
#
#
#    ##phase6 NON-context bp learning for entire network
#    #trainingSequence.startNextPhaseOfTraining
#    #phaseTrain { performStandardBackPropTrainingOn(neuronsWithInputLinks) }
#
#
#    forEachExampleDisplayInputsAndOutputs
#    testMSE = calcTestingMeanSquaredErrors
#    return trainingSequence.epochs, calcMSE, testMSE
#  end
#
#  def performStandardBackPropTrainingOn(learningNeurons)
#    zeroWeightsOfNon(learningNeurons)
#    acrossExamplesAccumulateDeltaWs(learningNeurons) { |aNeuron, dataRecord| aNeuron.calcDeltaWsAndAccumulate }
#    neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
#  end
#
#  def zeroWeightsOfNon(learningNeurons)
#    nonLearningNeurons = neuronsWithInputLinks - learningNeurons.flatten
#    nonLearningNeurons.each { |neuron| neuron.zeroWeights }
#  end
#end


###################################################################

#class WeightChangeNormalizer
#  attr_accessor :layer, :weightChangeSetPoint
#
#  def initialize(layer, args, weightChangeSetPoint = 0.08)
#    @layer = layer
#    @weightChangeSetPoint = weightChangeSetPoint
#  end
#
#  def normalizeWeightChanges
#    layerGain = weightChangeSetPoint / maxWeightChange
#    layer.each { |neuron| neuron.inputLinks.each { |aLink| aLink.deltaWAccumulated = layerGain * aLink.deltaWAccumulated } }
#  end
#
#  private
#
#  def maxWeightChange
#    acrossLayerMaxValues = layer.collect do |aNeuron|
#      accumulatedDeltaWsAbs = aNeuron.inputLinks.collect { |aLink| aLink.deltaWAccumulated.abs }
#      accumulatedDeltaWsAbs.max
#    end
#    return acrossLayerMaxValues.max
#  end
#end

