### VERSION "nCore"
; ## ../nCore/lib/core/TrainingBase.rb

require 'forwardable'

###################################################################
###################################################################
; ###### Forwarding To Learning Strategy from Neuron ##############

module ForwardingToLearningStrategy

  def startStrategy
    learningStrat.startStrategy
  end

  def startEpoch
    learningStrat.startEpoch
  end

  def propagate(exampleNumber)
    learningStrat.propagate(exampleNumber)
  end

  def learnExample
    learningStrat.learnExample
  end

  def endEpoch
    learningStrat.endEpoch
  end

  def finishLearningStrategy
    learningStrat.finishLearningStrategy
  end

  # service routines that may be used by various learning strategies

  def calcWeightsForUNNormalizedInputs
    learningStrat.calcWeightsForUNNormalizedInputs
  end

end

###################################################################
###################################################################
; ####### Neurons with Plug-in Learning Strategies ################

class Neuron2 < Neuron
  attr_accessor :learningStrat
  include ForwardingToLearningStrategy

  def postInitialize
    super
    @learningStrat = nil
  end

  def particularInits
  end

end

class OutputNeuron2 < OutputNeuron
  attr_accessor :learningStrat
  include ForwardingToLearningStrategy

  def postInitialize
    super
    @learningStrat = nil
  end

  def particularInits
  end

end

###################################################################
###################################################################
; ######################## Layer and LayerAry ######################

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

###############

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

###################################################################
###################################################################
; ####################### Learning Strategies #######################

class LearningStrategyBase # strategy for standard bp learning for output neurons
  attr_reader :neuron, :strategyArgs, :inputLinks, :outputLinks
  include CommonNeuronCalculations

  def initialize(theEnclosingNeuron, ** strategyArgs)
    @neuron = theEnclosingNeuron
    @strategyArgs = strategyArgs
    ioFunction = @strategyArgs[:ioFunction]
    self.extend(ioFunction)
    neuron.extend(ioFunction)
    neuron.output = neuron.ioFunction(neuron.netInput) # only needed for recurrent simulators
    @inputLinks = @neuron.inputLinks
    @outputLinks = @neuron.outputLinks if @neuron.respond_to?(:outputLinks)
  end

  def startStrategy
  end

  ## neuron.output = ioFunction(neuron.netInput) Simulating recurrent nets?
  def startEpoch
  end

  def learnExample
  end

  def endEpoch
  end

  def finishLearningStrategy
  end

  # service routines that may be used by various learning strategies
  def calcWeightsForUNNormalizedInputs
    biasWeight = inputLinks.inject(0.0) { |sum, link| sum + link.propagateUsingZeroInput }
    inputLinks.each { |aLink| aLink.calcWeightsForUNNormalizedInputs }
    inputLinks[-1].weight = biasWeight
    inputLinks.each { |aLink| aLink.resetAllNormalizationVariables }
  end
end

class ForwardPropOnly < LearningStrategyBase # just forward propagation
  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
  end
end

class LearningBP < LearningStrategyBase # strategy for standard bp learning for hidden neurons

  def startEpoch
    zeroDeltaWAccumulated
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
  end

  def learnExample
    neuron.error = calcNetError * ioDerivativeFromNetInput(neuron.netInput)
    calcDeltaWsAndAccumulate
  end

  def endEpoch
    addAccumulationToWeight
  end
end

class LearningBPOutput < LearningStrategyBase # strategy for standard bp learning for output neurons

  def startEpoch
    zeroDeltaWAccumulated
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron()
    neuron.output = output = ioFunction(netInput)
    neuron.target = target = neuron.arrayOfSelectedData[exampleNumber]
    neuron.outputError = output - target
  end

  def learnExample
    neuron.error = neuron.outputError * ioDerivativeFromNetInput(neuron.netInput)
    calcDeltaWsAndAccumulate
  end

  def endEpoch
    addAccumulationToWeight
  end
end

class Normalization < LearningStrategyBase

  def startEpoch
    inputLinks.each { |aLink| aLink.resetAllNormalizationVariables }
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
  end

  def learnExample
    inputLinks.each { |link| link.storeEpochHistory }
  end

  def endEpoch
    inputLinks.each { |aLink| aLink.calculateNormalizationCoefficients }
  end
end

class SelfOrgStrat < LearningStrategyBase
  attr_accessor :targetMinus, :distanceBetweenTargets

  def initialize(theEnclosingNeuron, ** strategyArgs)
    super
    @targetPlus = self.findNetInputThatGeneratesMaximumOutput
    @targetMinus = -1.0 * @targetPlus
    @distanceBetweenTargets = @targetPlus - @targetMinus
  end

  def startEpoch
    zeroDeltaWAccumulated
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
  end

  def learnExample
    calcSelfOrgError
    calcDeltaWsAndAccumulate
  end

  def endEpoch
    addAccumulationToWeight
  end

  private

  def calcSelfOrgError
    netInput = neuron.netInput
    neuron.error = -1.0 * neuron.ioDerivativeFromNetInput(netInput) * (((netInput - targetMinus)/distanceBetweenTargets) - 0.5)
  end
end

#########

module ContextForLearning
  attr_accessor :learningController

  def propagate(exampleNumber)
    if learningController.output == 1.0
      super(exampleNumber)
    else
      dropOutNeuron
    end
  end

  def learnExample
    if learningController.output == 1.0
      super
    end
  end

  protected

  def dropOutNeuron
    neuron.netInput = netInput = 0.0
    neuron.output = ioFunction(netInput)
  end
end

###################################################################
###################################################################
; ### Link Specialized for Normalization of Inputs ################

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
    self.weight = normalizationMultiplier * weight
  end

  def propagateUsingZeroInput
    return -1.0 * normalizationMultiplier * weight * normalizationOffset
  end

  def to_s
    return "Weight=\t#{weight}\tOffset=\t#{normalizationOffset}\tMultiplier=\t#{normalizationMultiplier}\tDeltaW=\t#{deltaW}\tAccumulatedDeltaW=\t#{deltaWAccumulated}\tWeightAtBeginningOfTraining=\t#{weightAtBeginningOfTraining}\tFROM: #{inputNeuron.class.to_s} #{inputNeuron.id} TO: #{outputNeuron.class.to_s} #{outputNeuron.id}"
  end
end

###################################################################
###################################################################
; ######################## Learning Controllers ####################

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

###################################################################
###################################################################
; #######################  DisplayAndErrorCalculations #############

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

###################################################################
###################################################################
; ######################## Training Sequencer ######################

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
; ######################## Base Trainer     ########################

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
    @trainingSequence = args[:trainingSequence].new(args)

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

###################################################################
########################################################
; ###################################################################

