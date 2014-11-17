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
      allNeuronLayers.propagateExample(exampleNumber)
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

  def initialize(input = [])
    @arrayOfNeurons = convertInputToArrayOfNeurons(input)
  end

  def_delegators :@arrayOfNeurons, :[], :size, :length, :each, :each_with_index, :collect, :all?

  def initWeights
    arrayOfNeurons.each { |aNeuron| aNeuron.initWeights }
  end

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

  def calcWeightsForUNNormalizedInputs
    arrayOfNeurons.each { |aNeuron| aNeuron.calcWeightsForUNNormalizedInputs }
  end

  def to_a
    return @arrayOfNeurons
  end

  def to_LayerAry
    LayerArray.new(self)
  end

  def <<(aNeuron)
    begin
      if aNeuron.kind_of?(NeuronBase)
        @arrayOfNeurons << aNeuron
        return
      end
      raise "ERROR: Attempting to append a NON-Neuron object to a Layer"
    rescue Exception => e
      puts e.message
      puts e.backtrace.inspect
    end
  end

  def convertInputToArrayOfNeurons(x)
    begin
      return x if (x.all? { |e| e.kind_of?(NeuronBase) }) # if x is an array of Neurons already!!
      return x if (x.length == 0) # if x is an empty array!!

      if (x.kind_of?(Array) && x.length == 1) # if x is an array of one array of Neurons already!!
        y = x[0]
        return y if (y.all? { |e| e.kind_of?(NeuronBase) })
      end

      if (x.kind_of?(LayerArray) && x.length == 1) # if x is a LayerArray with just one Layer within it!!
        return x[0].to_a
      end

      raise "Wrong Type of argument to initialize Layer: It is Not an Array of Neurons; nor an Array of an Array of Neurons; nor a Zero Length Array"
    rescue Exception => e
      puts e.message
      puts e.backtrace.inspect
    end
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
    @arrayOfLayers = convertInputToArrayOfLayers(arrayOfLayers)
  end

  def_delegators :@arrayOfLayers, :[], :size, :length, :each, :collect, :include?

  def initWeights
    arrayOfLayers.each { |aLayer| aLayer.initWeights }
  end

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

  def propagateAndLearnForAnEpoch(learningLayers, numberOfExamples)
    learningLayers.startEpoch
    numberOfExamples.times do |exampleNumber|
      propagateExample(exampleNumber)
      learningLayers.learnExample
    end
    learningLayers.endEpoch
  end

  def attachLearningStrategy(learningStrategy, strategyArgs)
    arrayOfLayers.each { |aLayer| aLayer.attachLearningStrategy(learningStrategy, strategyArgs) }
  end

  def calcWeightsForUNNormalizedInputs
    arrayOfLayers.each { |aLayer| aLayer.calcWeightsForUNNormalizedInputs }
  end

  def -(aLayerOraLayerArray)
    return LayerArray.new(arrayOfLayers - aLayerOraLayerArray.to_LayerAry.to_a)
  end

  def +(aLayer)
    self << aLayer
    return self
  end

  def <<(aLayer)
    begin
      if aLayer.kind_of?(Layer)
        @arrayOfLayers << aLayer
        return
      end
      raise "ERROR: Attempting to append a NON-Layer object to a LayerArray"
    rescue Exception => e
      STDERR.puts e.message
      STDERR.puts e.backtrace.inspect
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

  def convertInputToArrayOfLayers(x) # convert x to ARRAY of Layers
    begin
      return x if (x.all? { |e| e.kind_of?(Layer) }) # if array of array of Layers
      return [] if (x.length == 0) # if empty array
      return [x] if (x.kind_of?(Layer)) # if a single Layer

      x = [x] if (x.all? { |e| e.kind_of?(NeuronBase) }) # if single array neurons, then convert to array of array of neurons
      if (x.all? { |e| e.kind_of?(Array) }) # if array of array of neurons
        if (x.flatten.all? { |e| e.kind_of?(NeuronBase) })
          return x.collect { |e| e.to_Layer }
        end
      end
      raise "Wrong Type: It is Not an Array of Layers or Neurons; nor a Zero Length Array"
    rescue Exception => e
      puts e.message
      puts e.backtrace.inspect
    end
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

    @allNeuronLayers = network.allNeuronLayers.to_LayerAry
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

    learningLayers = allNeuronLayers - inputLayer
    propagatingLayers, controllingLayers = layerDetermination(learningLayers)

    strategyArgs = {:ioFunction => SigmoidIOFunction}
    hiddenLayers = learningLayers - outputLayer
    hiddenLayers.attachLearningStrategy(LearningBP, strategyArgs)
    outputLayer.attachLearningStrategy(LearningBPOutput, strategyArgs)
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
      propagatingLayers.propagateAndLearnForAnEpoch(learningLayers, numberOfExamples)
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
  end
end

module SelfOrg

  def selOrgNoContext(learningLayers, ioFunction, epochsDuringPhase, totalEpochs)
    propagatingLayers, willNotUseControllingLayer = layerDetermination(learningLayers.to_LayerAry)
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
end

module ForwardPropWithContext

  def forwardPropWithContext(layerReceivingContext, ioFunction)
    propagatingLayers, controllingLayers = layerDetermination(layerReceivingContext.to_LayerAry)
    singleLayerControllingLearning = controllingLayers[-1]
    strategyArguments = {:ioFunction => ioFunction}
    setupForwardPropWithContext(singleLayerControllingLearning, layerReceivingContext, strategyArguments)
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
    learningLayers = learningLayers.to_LayerAry
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
    learningLayers = hiddenLayer1
    learningLayers.initWeights # Needed only when the given layer is self-organizing for the first time
    mse, totalEpochs = selOrgNoContext(learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    ### self-org 2nd hidden layer WITH CONTEXT!!
    learningLayers = hiddenLayer2
    learningLayers.initWeights # Needed only when the given layer is self-organizing for the first time
    mse, totalEpochs = normalizationAndSelfOrgWithContext(learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    mse, totalEpochs = normalizationAndSelfOrgWithContext(learningLayers, ioFunction, 1, totalEpochs)
    puts "Hidden Layer 2 with effectively NO Learning but with Outputs in Context (i.e., with 'dont know' representation added back)"
    forEachExampleDisplayInputsAndOutputs(hiddenLayer2)

    layersThatWereNormalized = [hiddenLayer1, hiddenLayer2].to_LayerAry
    layersThatWereNormalized.calcWeightsForUNNormalizedInputs # for understanding, convert to normal neural weight representation (without normalization variables)

    learningLayers = outputLayer.to_LayerAry
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
    learningLayers = hiddenLayer1
    learningLayers.initWeights
    mse, totalEpochs = selOrgNoContext(learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    ### self-org 2nd hidden layer WITH CONTEXT!!
    learningLayers = hiddenLayer2
    learningLayers.initWeights
    mse, totalEpochs = normalizationAndSelfOrgWithContext(learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    ### self-org 2nd hidden layer withOUT context!!
    mse, totalEpochs = selOrgNoContext(learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    layerReceivingContext = hiddenLayer2
    forwardPropWithContext(layerReceivingContext, ioFunction)
    puts "Hidden Layer 2 with effectively NO Learning but with Outputs in Context (i.e., with 'dont know' representation added back)"
    forEachExampleDisplayInputsAndOutputs(hiddenLayer2)

    layersThatWereNormalized = [hiddenLayer1, hiddenLayer2].to_LayerAry
    layersThatWereNormalized.calcWeightsForUNNormalizedInputs

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

