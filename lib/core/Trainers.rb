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
    distributeSetOfExamples(examples)
    totalEpochs = 0

    learningLayers = allNeuronLayers - [inputLayer]
    propagatingLayers, controllingLayers = layerDetermination(learningLayers)

    strategyArgs = {:ioFunction => SigmoidIOFunction}
    attachLearningStrategy(learningLayers - [outputLayer], LearningBP, strategyArgs)
    attachLearningStrategy([outputLayer], LearningBPOutput, strategyArgs)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)

    forEachExampleDisplayInputsAndOutputs
    return totalEpochs, mse, calcTestingMeanSquaredErrors
  end

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
  end

  def trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
    mse = 1e100

    startStrategy(learningLayers)
    while ((mse >= minMSE) && trainingSequence.stillMoreEpochs)
      propagateAndLearnForAnEpoch(propagatingLayers, learningLayers)
      trainingSequence.nextEpoch
      mse = calcMeanSumSquaredErrors if (entireNetworkSetup?)

      currentEpochNumber = trainingSequence.epochs + totalEpochs
      puts "current epoch number= #{currentEpochNumber}\tmse = #{mse}" if (currentEpochNumber % 100 == 0)
    end

    totalEpochs += trainingSequence.epochs
    trainingSequence.startNextPhaseOfTraining
    return mse, totalEpochs
  end

  def startStrategy(learningLayers)
    learningLayers.each do |aLayerOfNeurons|
      aLayerOfNeurons.each {|aNeuron| aNeuron.startStrategy}
    end
  end

  def entireNetworkSetup?
    allNeuronsThatCanHaveLearningStrategy = (allNeuronLayers[1..-1]).flatten
    arrayThatMayIncludeNils = allNeuronsThatCanHaveLearningStrategy.collect { |neuron| neuron.learningStrat }
    !arrayThatMayIncludeNils.include?(nil)
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

  ###################################################################
  ###------------  Core Metric Support Section -------------------###

  def layerDetermination(learningLayers)
    lastLearningLayer = learningLayers[-1]
    propagatingLayers = []
    controllingLayer = nil

    allNeuronLayers.each do |aLayer|
      propagatingLayers << aLayer
      break if aLayer == lastLearningLayer
      controllingLayer = aLayer
    end

    controllingLayers = [controllingLayer]
    return [propagatingLayers, controllingLayers]
  end

  def attachLearningStrategy(layers, learningStrategy, strategyArgs)
    layers.each do |neurons|
      neurons.each { |neuron| neuron.learningStrat = learningStrategy.new(neuron, strategyArgs) }
    end
  end

  def calcMeanSumSquaredErrors # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder
    genericCalcMeanSumSquaredErrors(numberOfExamples)
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

  def forEachExampleDisplayInputsAndOutputs(resultsLayer = outputLayer)
    breakOnNextPass = false
    propagatingLayers = allNeuronLayers.collect do |aLayer|
      next if breakOnNextPass
      breakOnNextPass = true if (aLayer == resultsLayer)
      aLayer
    end
    propagatingLayers.compact!
    #
    examples.each_with_index do |anExample, exampleNumber|
      inputs = anExample[:inputs]
      propagateExampleAcross(propagatingLayers, exampleNumber)
      results = resultsLayer.collect { |aResultsNeuron| aResultsNeuron.output }
      puts "\t\t\tinputs= #{inputs}\tresults= #{results}"
    end
  end

  def propagateAcrossEntireNetwork(exampleNumber)
    propagateExampleAcross(allNeuronLayers, exampleNumber)
    #allNeuronsInOneArray.flatten.each { |aNeuron| aNeuron.propagate(exampleNumber) }
  end

  def calcWeightedErrorMetricForExample
    outputLayer.collect { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
  end

end

class Trainer2SelfOrgAndContext < TrainerBase

  attr_accessor :hiddenLayer1, :hiddenLayer2

  def postInitialize
    @hiddenLayer1 = allNeuronLayers[1]
    @hiddenLayer2 = allNeuronLayers[2]
  end

  def train
    distributeSetOfExamples(examples)

    totalEpochs = 0
    ioFunction = NonMonotonicIOFunction

    ### Now will self-org 1st hidden layer
    learningLayers = [hiddenLayer1]
    totalEpochs, mse = simplifiedSelfOrg(learningLayers, ioFunction, totalEpochs)


    ### Now will self-org 2nd hidden layer
    learningLayers = [hiddenLayer2]
    totalEpochs, mse = simplifiedSelfOrg(learningLayers, ioFunction, totalEpochs)

    ## TODO what's the value in doing this?  -- apparently NOT!
    layersThatWereNormalized = [hiddenLayer1, hiddenLayer2]
    calcWeightsForUNNormalizedInputs(layersThatWereNormalized)

    #forEachExampleDisplayInputsAndOutputs
    return totalEpochs, mse, 0.998 # calcTestingMeanSquaredErrors
  end

  def simplifiedSelfOrg(learningLayers, ioFunction, totalEpochs)

    propagatingLayers, controllingLayers = layerDetermination(learningLayers)
    controllingLayer = controllingLayers[0]

    strategyArguments = {}
    strategyArguments[:ioFunction] = ioFunction

    if (controllingLayer == inputLayer)
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

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer])
  end

  def calcWeightsForUNNormalizedInputs(learningLayers)
    learningLayers.each { |neurons| neurons.each { |aNeuron| aNeuron.calcWeightsForUNNormalizedInputs } }
  end
end


class Trainer3SelfOrgContextSuper < Trainer2SelfOrgAndContext

  def train
    distributeSetOfExamples(examples)

    totalEpochs = 0
    ioFunction = NonMonotonicIOFunction

    ### Now will self-org 1st hidden layer
    learningLayers = [hiddenLayer1]
    totalEpochs, mse = simplifiedSelfOrg(learningLayers, ioFunction, totalEpochs)

    ### Now will self-org 2nd hidden layer  WITH CONTEXT!!
    learningLayers = [hiddenLayer2]
    totalEpochs, mse = simplifiedSelfOrg(learningLayers, ioFunction, totalEpochs)


    ## TODO what's the value in doing this?  -- apparently NOT!
    layersThatWereNormalized = [hiddenLayer1, hiddenLayer2]
    calcWeightsForUNNormalizedInputs(layersThatWereNormalized)

    learningLayers = [outputLayer]
    totalEpochs, mse = supervisedTraining(learningLayers, ioFunction, totalEpochs)

    forEachExampleDisplayInputsAndOutputs(outputLayer)

    return totalEpochs, mse, calcTestingMeanSquaredErrors
  end

  def supervisedTraining(learningLayers, ioFunction, totalEpochs)
    propagatingLayers, controllingLayers = layerDetermination(learningLayers)

    strategyArguments = {}
    strategyArguments[:ioFunction] = ioFunction
    attachLearningStrategy(learningLayers, LearningBPOutput, strategyArguments) if learningLayers.include?(outputLayer)

    otherLearningLayers = learningLayers - [outputLayer]
    attachLearningStrategy(otherLearningLayers, LearningBP, strategyArguments)

    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, totalEpochs)
    return totalEpochs, mse
  end

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
  end
end

class Trainer4SelfOrgContextSuper < Trainer3SelfOrgContextSuper

  def train
    distributeSetOfExamples(examples)

    totalEpochs = 0
    ioFunction = NonMonotonicIOFunction

    ### Now will self-org 1st hidden layer
    learningLayers = [hiddenLayer1]
    totalEpochs, mse = simplifiedSelfOrg(learningLayers, ioFunction, totalEpochs)

    ### Now will self-org 2nd hidden layer WITH CONTEXT!!
    learningLayers = [hiddenLayer2]
    totalEpochs, mse = simplifiedSelfOrg(learningLayers, ioFunction, totalEpochs)

    ### Now will self-org 2nd hidden layer withOUT context!!
    learningLayers = [hiddenLayer2]
    propagatingLayers, dumbdumb = layerDetermination(learningLayers)
    strategyArguments = {}
    strategyArguments[:ioFunction] = ioFunction
    # TODO return arguments below are in reverse order
    mse, totalEpochs = normalizationAndSelfOrgWITHOUTContext(learningLayers, propagatingLayers, strategyArguments, totalEpochs)

    # TODO what's the value in doing this?
    layersThatWereNormalized = [hiddenLayer1, hiddenLayer2]
    calcWeightsForUNNormalizedInputs(layersThatWereNormalized)

    learningLayers = [outputLayer]
    # learningLayers = [hiddenLayer1, hiddenLayer2, outputLayer]
    totalEpochs, mse = supervisedTraining(learningLayers, ioFunction, totalEpochs)

    forEachExampleDisplayInputsAndOutputs(outputLayer)

    return totalEpochs, mse, calcTestingMeanSquaredErrors
  end
end


class TrainerAD1 < Trainer3SelfOrgContextSuper


end


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

