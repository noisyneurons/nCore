### VERSION "nCore"
## ../nCore/lib/core/Trainers.rb

require 'forwardable'

###################################################################
###################################################################

class TrainingSequence
  attr_accessor :args, :epochs, :maxNumberOfEpochs,
                :stillMoreEpochs

  def initialize(args)
    @args = args
    @epochs = -1
    @maxNumberOfEpochs = 0 # need to do this to enable first 'nextEpoch' to work
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

  def startTrainingPhase(epochsDuringPhase)
    self.maxNumberOfEpochs = epochsDuringPhase
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

###################################################################
###################################################################

module DisplayAndErrorCalculations

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

  def calcWeightedErrorMetricForExample
    outputLayer.collect { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
  end

  def forEachExampleDisplayNetworkInputsAndNetInputTo(resultsLayer = outputLayer)
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
      results = resultsLayer.collect { |aResultsNeuron| aResultsNeuron.netInput }
      puts "\t\t\tinputs= #{inputs}\tresults= #{results}"
    end
  end

  def forEachExampleDisplayInputsAndOutputs(resultsLayer = outputLayer)
    breakOnNextPass = false
    propagatingLayers = allNeuronLayers.collect do |aLayer|
      next if breakOnNextPass
      breakOnNextPass = true if (aLayer == resultsLayer)
      aLayer
    end
    propagatingLayers = propagatingLayers.compact.to_LayerAry
    #
    examples.each_with_index do |anExample, exampleNumber|
      inputs = anExample[:inputs]
      propagatingLayers.propagateExample(exampleNumber)
      results = resultsLayer.collect { |aResultsNeuron| aResultsNeuron.output }
      puts "\t\t\tinputs= #{inputs}\tresults= #{results}"
    end
  end
end


class Layer
  attr_reader :arrayOfNeurons
  extend Forwardable

  def initialize(anArrayOfNeurons = [])
    @arrayOfNeurons = standardizeInputFormat(anArrayOfNeurons)
  end

  def_delegators :@arrayOfNeurons, :[], :size, :length, :each, :each_with_index, :collect, :all?

  def startStrategy
    arrayOfNeurons.each { |aNeuron| aNeuron.startStrategy }
  end

  def startEpoch
    arrayOfNeurons.each { |aNeuron| aNeuron.startEpoch }
  end

  def propagateExample(exampleNumber)
    arrayOfNeurons.each { |aNeuron| aNeuron.propagate(exampleNumber) }
  end

  def learnExample
    arrayOfNeurons.each { |aNeuron| aNeuron.learnExample }
  end

  def endEpoch
    arrayOfNeurons.each { |aNeuron| aNeuron.endEpoch }
  end

  def attachLearningStrategy(learningStrategy, strategyArgs)
    arrayOfNeurons.each { |aNeuron| aNeuron.learningStrat = learningStrategy.new(aNeuron, strategyArgs) }
  end

  def to_a
    return @arrayOfNeurons
  end

  def to_LayerAry
    LayerArray.new(self)
  end

  def standardizeInputFormat(x)
    return x if (x.length == 0)
    return x if (x.all? { |e| e.kind_of?(NeuronBase) })
    STDERR.puts "Wrong Type: It is Not an Array of Neurons; nor a Zero Length Array"
  end

  def setup?
    statusAry = arrayOfNeurons.collect { |aNeuron| aNeuron.learningStrat }
    !statusAry.include?(nil)
  end
end

class LayerArray
  attr_reader :arrayOfLayers
  extend Forwardable

  def initialize(arrayOfLayers=[])
    @arrayOfLayers = standardizeInputFormat(arrayOfLayers)
  end

  def_delegators :@arrayOfLayers, :[], :size, :length, :each, :collect, :include?

  def startStrategy
    arrayOfLayers.each { |aLayer| aLayer.startStrategy }
  end

  def startEpoch
    arrayOfLayers.each { |aLayer| aLayer.startEpoch }
  end

  def propagateExample(exampleNumber)
    arrayOfLayers.each { |aLayer| aLayer.propagateExample(exampleNumber) }
  end

  def learnExample
    arrayOfLayers.reverse.each { |aLayer| aLayer.learnExample }
  end

  def endEpoch
    arrayOfLayers.each { |aLayer| aLayer.endEpoch }
  end

  def attachLearningStrategy(learningStrategy, strategyArgs)
    arrayOfLayers.each { |aLayer| aLayer.attachLearningStrategy(learningStrategy, strategyArgs) }
  end

  def -(aLayerOraLayerArray)
    return LayerArray.new(arrayOfLayers - stdFormat(aLayerOraLayerArray))
  end

  def +(aLayer)
    self << aLayer
    return self
  end

  def <<(aLayer)
    if aLayer.kind_of?(Layer)
      @arrayOfLayers << aLayer
    else
      STDERR.puts "ERROR: Attempting to append a NON-Layer object to a LayerArray; The object is #{aLayer}"
    end
  end

  def to_a
    return arrayOfLayers
  end

  def to_LayerAry
    return self
  end

  def setup?
    statusAry = arrayOfLayers.collect { |aLayer| aLayer.setup? }
    !statusAry.include?(false)
  end

  def standardizeInputFormat(x)
    return x if (x.length == 0)
    return x if (x.all? { |e| e.kind_of?(Layer) })
    return [x] if (x.all? { |e| e.kind_of?(NeuronBase) })
    STDERR.puts "Wrong Type: It is Not an Array of Layers or Neurons; nor a Zero Length Array"
  end

  def stdFormat(aLayerOraLayerArray)
    return [aLayerOraLayerArray] if aLayerOraLayerArray.kind_of?(Layer)
    return aLayerOraLayerArray.to_a if aLayerOraLayerArray.kind_of?(LayerArray)
    STDERR.puts "ERROR: Attempting to 'delete' a NON-Layer object from a LayerArray"
  end
end

class TrainerBase
  attr_accessor :examples, :network, :numberOfOutputNeurons, :allNeuronLayers, :inputLayer,
                :outputLayer, :theBiasNeuron, :args, :trainingSequence, :numberOfExamples,
                :startTime, :elapsedTime, :minMSE
  include NeuronToNeuronConnection, ExampleDistribution, DisplayAndErrorCalculations

  def initialize(examples, network, args)
    @args = args
    @network = network

    ary = network.allNeuronLayers
    aryOfLayers = ary.collect { |e| e.to_Layer }
    @allNeuronLayers = aryOfLayers.to_LayerAry
    @inputLayer = @allNeuronLayers[0]
    @outputLayer = @allNeuronLayers[-1]
    @theBiasNeuron = network.theBiasNeuron
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
    hiddenLayers = learningLayers - [outputLayer]
    hiddenLayers.attachLearningStrategy(LearningBP, strategyArgs)
    layersOfOutputNeurons = [outputLayer].to_nAry
    layersOfOutputNeurons.attachLearningStrategy(LearningBPOutput, strategyArgs)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, epochsDuringPhase=2000, totalEpochs)

    forEachExampleDisplayInputsAndOutputs
    return totalEpochs, mse, calcTestingMeanSquaredErrors
  end

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
  end

  def layerDetermination(learningLayers)
    lastLearningLayer = learningLayers[-1]
    propagatingLayers = LayerArray.new
    controllingLayers = LayerArray.new

    allNeuronLayers.each do |aLayer|
      propagatingLayers << aLayer
      break if aLayer == lastLearningLayer
      controllingLayers << aLayer
    end

    return [propagatingLayers, controllingLayers]
  end

  def trainingPhaseFor(propagatingLayers, learningLayers, epochsDuringPhase, totalEpochs)

    learningLayers.startStrategy

    mse = 1e100
    trainingSequence.startTrainingPhase(epochsDuringPhase)

    while ((mse >= minMSE) && trainingSequence.stillMoreEpochs)
      propagateAndLearnForAnEpoch(propagatingLayers, learningLayers)
      trainingSequence.nextEpoch
      mse = calcMeanSumSquaredErrors if (entireNetworkSetup?)

      currentEpochNumber = trainingSequence.epochs + totalEpochs
      puts "current epoch number= #{currentEpochNumber}\tmse = #{mse}" if (currentEpochNumber % 100 == 0)
    end

    totalEpochs += trainingSequence.epochs
    return mse, totalEpochs
  end

  def entireNetworkSetup?
    aLayerArray = allNeuronLayers[1..-1].to_LayerAry
    return aLayerArray.setup?
    #allNeuronsThatCanHaveLearningStrategy = (allNeuronLayers[1..-1]).flatten
    #arrayThatMayIncludeNils = allNeuronsThatCanHaveLearningStrategy.collect { |neuron| neuron.learningStrat }
    #!arrayThatMayIncludeNils.include?(nil)
  end

  def propagateAndLearnForAnEpoch(propagatingLayers, learningLayers)
    learningLayers.startEpoch
    numberOfExamples.times do |exampleNumber|
      propagatingLayers.propagateExample(exampleNumber)
      learningLayers.learnExample
    end
    learningLayers.endEpoch
  end

  def propagateAcrossEntireNetwork(exampleNumber)
    allNeuronLayers.propagateExample(exampleNumber)
  end
end

module SelfOrg

  def selOrgNoContext(learningLayers, ioFunction, epochsDuringPhase, totalEpochs)
    propagatingLayers, willNotUseControllingLayer = layerDetermination(learningLayers)
    strategyArguments = {:ioFunction => ioFunction}
    mse, totalEpochs = normalizationAndSelfOrgNoContext(learningLayers, propagatingLayers, strategyArguments, epochsDuringPhase, totalEpochs)
    return mse, totalEpochs
  end

  def normalizationAndSelfOrgNoContext(learningLayers, propagatingLayers, strategyArguments, epochsForSelfOrg, totalEpochs)
    learningLayers.attachLearningStrategy(Normalization, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, epochsForNormalization=1, totalEpochs)

    learningLayers.attachLearningStrategy(SelfOrgStrat, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, epochsForSelfOrg, totalEpochs)
    return mse, totalEpochs
  end

  def initWeights(learningLayers)
    learningLayers.each do |aLayer|
      aLayer.each { |aNeuron| initNeuronsWeights(aNeuron) }
    end
  end

  def initNeuronsWeights(neuron)
    inputLinks = neuron.inputLinks
    numberOfInputsToNeuron = inputLinks.length
    inputLinks.each do |aLink|
      verySmallNoise = 0.0001 * (rand - 0.5)
      weight = (0.2 + verySmallNoise) / numberOfInputsToNeuron # TODO may want sqrt(numberOfInputsToNeuron)
      aLink.weight = weight
      aLink.weightAtBeginningOfTraining = weight
    end
  end

  def calcWeightsForUNNormalizedInputs(learningLayers)
    learningLayers.each { |neurons| neurons.each { |aNeuron| aNeuron.calcWeightsForUNNormalizedInputs } }
  end
end

module ForwardPropWithContext

  def forwardPropWithContext(layerReceivingContext, ioFunction)
    propagatingLayers, controllingLayers = layerDetermination(layerReceivingContext)
    singleLayerControllingLearning = controllingLayers[-1]
    singleLayerReceivingContext = layerReceivingContext[0]
    strategyArguments = {:ioFunction => ioFunction}
    setupForwardPropWithContext(singleLayerControllingLearning, singleLayerReceivingContext, strategyArguments)
  end

  def setupForwardPropWithContext(singleLayerControllingLearning, singleLayerReceivingContext, strategyArguments)
    singleLayerControllingLearning.each_with_index do |neuronInControllingLayer, indexToControlNeuron|
      indexToLearningNeuron = 2 * indexToControlNeuron

      forwardPropWithContextSetup(LearningControlledByNeuron, indexToLearningNeuron, neuronInControllingLayer,
                                  singleLayerReceivingContext, strategyArguments)
      forwardPropWithContextSetup(LearningControlledByNeuronNOT, (indexToLearningNeuron + 1), neuronInControllingLayer,
                                  singleLayerReceivingContext, strategyArguments)
    end
  end

  def forwardPropWithContextSetup(classOfLearningController, indexToLearningNeuron, neuronInControllingLayer,
      singleLearningLayer, strategyArguments)
    aLearningNeuron = singleLearningLayer[indexToLearningNeuron]
    strategy = ForwardPropOnly.new(aLearningNeuron, strategyArguments)
    strategy.extend(ContextForLearning)
    learningController = classOfLearningController.new(neuronInControllingLayer)
    strategy.learningController = learningController
    aLearningNeuron.learningStrat = strategy
  end

end

module SelfOrgWithContext

  def normalizationAndSelfOrgWithContext(learningLayers, ioFunction, epochsForSelfOrg, totalEpochs) # (learningLayers, controllingLayers, propagatingLayers, strategyArguments, epochsForSelfOrg, totalEpochs)

    propagatingLayers, controllingLayers = layerDetermination(learningLayers)
    singleLayerControllingLearning = controllingLayers[-1]
    singleLearningLayer = learningLayers[0]

    strategyArguments = {:ioFunction => ioFunction}

    setupNormalizationStrategyWithContext(singleLayerControllingLearning, singleLearningLayer, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, epochsForNormalization=1, totalEpochs)

    setupSelfOrgStrategyWithContext(singleLayerControllingLearning, singleLearningLayer, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, epochsForSelfOrg, totalEpochs)
    return mse, totalEpochs
  end

  def setupNormalizationStrategyWithContext(singleLayerControllingLearning, singleLearningLayer, strategyArguments)

    singleLayerControllingLearning.each_with_index do |neuronInControllingLayer, indexToControlNeuron|
      indexToLearningNeuron = 2 * indexToControlNeuron

      normalizationSetup(LearningControlledByNeuron, indexToLearningNeuron, neuronInControllingLayer,
                         singleLearningLayer, strategyArguments)

      indexToLearningNeuron = indexToLearningNeuron + 1
      normalizationSetup(LearningControlledByNeuronNOT, indexToLearningNeuron, neuronInControllingLayer,
                         singleLearningLayer, strategyArguments)
    end
  end

  def normalizationSetup(classOfLearningController, indexToLearningNeuron, neuronInControllingLayer,
      singleLearningLayer, strategyArguments)

    aLearningNeuron = singleLearningLayer[indexToLearningNeuron]
    strategy = Normalization.new(aLearningNeuron, strategyArguments)
    strategy.extend(ContextForLearning)
    learningController = classOfLearningController.new(neuronInControllingLayer)
    strategy.learningController = learningController
    aLearningNeuron.learningStrat = strategy
  end

  def setupSelfOrgStrategyWithContext(singleLayerControllingLearning, singleLearningLayer, strategyArguments)
    singleLayerControllingLearning.each_with_index do |neuronInControllingLayer, indexToControlNeuron|
      indexToLearningNeuron = 2 * indexToControlNeuron

      selfOrgSetup(LearningControlledByNeuron, indexToLearningNeuron, neuronInControllingLayer,
                   singleLearningLayer, strategyArguments)
      selfOrgSetup(LearningControlledByNeuronNOT, (indexToLearningNeuron + 1), neuronInControllingLayer,
                   singleLearningLayer, strategyArguments)
    end
  end

  def selfOrgSetup(classOfLearningController, indexToLearningNeuron, neuronInControllingLayer,
      singleLearningLayer, strategyArguments)
    aLearningNeuron = singleLearningLayer[indexToLearningNeuron]
    strategy = SelfOrgStrat.new(aLearningNeuron, strategyArguments)
    strategy.extend(ContextForLearning)
    learningController = classOfLearningController.new(neuronInControllingLayer)
    strategy.learningController = learningController
    aLearningNeuron.learningStrat = strategy
  end

end


class Trainer3SelfOrgContextSuper < TrainerBase
  attr_accessor :hiddenLayer1, :hiddenLayer2
  include SelfOrg
  include SelfOrgWithContext

  def postInitialize
    @hiddenLayer1 = allNeuronLayers[1]
    @hiddenLayer2 = allNeuronLayers[2]
  end

  def train
    distributeSetOfExamples(examples)

    totalEpochs = 0
    ioFunction = NonMonotonicIOFunction

    ### self-org 1st hidden layer
    learningLayers = [hiddenLayer1].to_nAry
    initWeights(learningLayers) # Needed only when the given layer is self-organizing for the first time
    mse, totalEpochs = selOrgNoContext(learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    ### self-org 2nd hidden layer WITH CONTEXT!!
    learningLayers = [hiddenLayer2].to_nAry
    initWeights(learningLayers) # Needed only when the given layer is self-organizing for the first time
    mse, totalEpochs = normalizationAndSelfOrgWithContext(learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    mse, totalEpochs = normalizationAndSelfOrgWithContext(learningLayers, ioFunction, 1, totalEpochs)
    puts "Hidden Layer 2 with effectively NO Learning but with Outputs in Context (i.e., with 'dont know' representation added back)"
    forEachExampleDisplayInputsAndOutputs(hiddenLayer2)

    layersThatWereNormalized = [hiddenLayer1, hiddenLayer2]
    calcWeightsForUNNormalizedInputs(layersThatWereNormalized) # for understanding, convert to normal neural weight representation (without normalization variables)

    learningLayers = [outputLayer].to_nAry
    mse, totalEpochs = supervisedTraining(learningLayers, ioFunction, args[:epochsForSupervisedTraining], totalEpochs)

    return totalEpochs, mse, calcTestingMeanSquaredErrors
  end

  def supervisedTraining(learningLayers, ioFunction, epochsDuringPhase, totalEpochs)
    propagatingLayers, controllingLayers = layerDetermination(learningLayers)

    strategyArguments = {:ioFunction => ioFunction}
    learningLayers.attachLearningStrategy(LearningBPOutput, strategyArguments) if learningLayers.include?(outputLayer)

    otherLearningLayers = learningLayers - outputLayer
    otherLearningLayers.attachLearningStrategy(LearningBP, strategyArguments)

    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, epochsDuringPhase, totalEpochs)
    return mse, totalEpochs
  end

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
  end
end

class Trainer4SelfOrgContextSuper < Trainer3SelfOrgContextSuper
  include ForwardPropWithContext

  def train
    distributeSetOfExamples(examples)

    totalEpochs = 0
    ioFunction = NonMonotonicIOFunction

    ### self-org 1st hidden layer
    learningLayers = hiddenLayer1.to_LayerAry
    initWeights(learningLayers)
    mse, totalEpochs = selOrgNoContext(learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    ### self-org 2nd hidden layer WITH CONTEXT!!
    learningLayers = hiddenLayer2.to_LayerAry
    initWeights(learningLayers)
    mse, totalEpochs = normalizationAndSelfOrgWithContext(learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    ### self-org 2nd hidden layer withOUT context!!
    mse, totalEpochs = selOrgNoContext(learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    layersReceivingContext= hiddenLayer2.to_LayerAry
    forwardPropWithContext(layersReceivingContext, ioFunction)
    puts "Hidden Layer 2 with effectively NO Learning but with Outputs in Context (i.e., with 'dont know' representation added back)"
    forEachExampleDisplayInputsAndOutputs(hiddenLayer2)

    layersThatWereNormalized = [hiddenLayer1, hiddenLayer2]
    calcWeightsForUNNormalizedInputs(layersThatWereNormalized)

    learningLayers = outputLayer.to_LayerAry
    mse, totalEpochs = supervisedTraining(learningLayers, ioFunction, args[:epochsForSupervisedTraining], totalEpochs)

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

