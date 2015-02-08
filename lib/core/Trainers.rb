### VERSION "nCore"
; ## ../nCore/lib/core/Trainers.rb

###################################################################
###################################################################
; ################# MODULE: DisplayAndErrorCalculations #############

module DisplayAndErrorCalculations

  def calcMeanSumSquaredErrors # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder
    mse = 1e100
    mse = genericCalcMeanSumSquaredErrors(numberOfExamples) unless (outputLayer[0].outputError.nil?)
    return mse
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
      allNeuronLayers.propagate(exampleNumber)
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
      logger.puts "\t\t\tinputs= #{inputs}\tresults= #{results}"
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
      propagatingLayers.propagate(exampleNumber)
      results = resultsLayer.collect { |aResultsNeuron| aResultsNeuron.output }
      logger.puts "\t\t\tinputs= #{inputs}\tresults= #{results}"
    end
  end
end

###################################################################
###################################################################
; ######################## Base Trainer  #############################

class TrainerBase
  attr_accessor :examples, :network, :numberOfOutputNeurons, :allNeuronLayers, :inputLayer,
                :outputLayer, :theBiasNeuron, :args, :trainingSequence, :numberOfExamples,
                :startTime, :elapsedTime, :minMSE, :logger
  include NeuronToNeuronConnection, ExampleDistribution, DisplayAndErrorCalculations

  def initialize(examples, network, args)
    @args = args
    @logger = @args[:logger]
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
    learningLayers.attachLearningStrategy(LearningBP, strategyArgs)
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
      mse = calcMeanSumSquaredErrors

      currentEpochNumber = trainingSequence.epochs + totalEpochs
      logger.puts "current epoch number= #{currentEpochNumber}\tmse = #{mse}" if (currentEpochNumber % 100 == 0)
    end

    totalEpochs += trainingSequence.epochs
    return mse, totalEpochs
  end
end

###################################################################
###################################################################
; ########### MODULES: SelfOrg and ForwardProp modules #############

#module ForwardPropWithContext
#
#  def forwardPropWithContext(layerReceivingContext, ioFunction)
#    propagatingLayers, controllingLayers = layerDetermination(layerReceivingContext.to_LayerAry)
#    singleLayerControllingLearning = controllingLayers[-1]
#    strategyArguments = {:ioFunction => ioFunction}
#    setupForwardPropWithContext(singleLayerControllingLearning, layerReceivingContext, strategyArguments)
#  end
#
#  def setupForwardPropWithContext(singleLayerControllingLearning, singleLayerReceivingContext, strategyArguments)
#    singleLayerControllingLearning.each_with_index do |neuronInControllingLayer, indexToControlNeuron|
#      indexToLearningNeuron = 2 * indexToControlNeuron
#
#      forwardPropWithContextSetup(LearningControlledByNeuron, indexToLearningNeuron, neuronInControllingLayer,
#                                  singleLayerReceivingContext, strategyArguments)
#      forwardPropWithContextSetup(LearningControlledByNeuronNOT, (indexToLearningNeuron + 1), neuronInControllingLayer,
#                                  singleLayerReceivingContext, strategyArguments)
#    end
#  end
#
#  def forwardPropWithContextSetup(classOfLearningController, indexToLearningNeuron, neuronInControllingLayer,
#      singleLearningLayer, strategyArguments)
#    aLearningNeuron = singleLearningLayer[indexToLearningNeuron]
#    strategy = ForwardPropOnly.new(aLearningNeuron, strategyArguments)
#    strategy.extend(ContextForLearning)
#    learningController = classOfLearningController.new(neuronInControllingLayer)
#    strategy.learningController = learningController
#    aLearningNeuron.learningStrat = strategy
#  end
#
#end

module SelfOrgMixture
  def selfOrgNoContextStrategyArgs
    {
        :classOfInputDistributionModel => ExampleDistributionModel, :desiredMeanNetInput => 3.0,
        :ioFunction => SigmoidIOFunction, :extendStrategyWithModule => nil
    }
  end

  def selOrg(strategyArgs, learningLayers, ioFunction, epochsDuringPhase, totalEpochs)
    strategyArgs[:ioFunction] = ioFunction
    propagatingLayers, notUsed = layerDetermination(learningLayers.to_LayerAry)
    mse, totalEpochs = selfOrgTraining(learningLayers, strategyArgs, propagatingLayers, epochsDuringPhase, totalEpochs)
    learningLayers.attachLearningStrategy(ForwardPropOnly, strategyArgs)
    return mse, totalEpochs
  end

  def selfOrgTraining(learningLayers, strategyArguments, propagatingLayers, epochsForSelfOrg, totalEpochs)
    iterations = (epochsForSelfOrg - 4)/2

    #puts (propagatingLayers - inputLayer)
    #(propagatingLayers - inputLayer).attachLearningStrategy(ForwardPropOnly, strategyArguments)

    learningLayers.attachLearningStrategy(NormalizeByZeroingSumOfNetInputs, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, 1, totalEpochs)

    learningLayers.attachLearningStrategy(ScaleNeuronWeights, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, 1, totalEpochs)

    iterations.times do |i|
      learningLayers.attachLearningStrategy(EstimateInputDistribution, strategyArguments)
      mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, epochsForEstimation=1, totalEpochs)

      learningLayers.attachLearningStrategy(SelfOrgByContractingBothLobesOfDistribution, strategyArguments)
      mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, epochsForAdapting=1, totalEpochs)
    end

    learningLayers.attachLearningStrategy(MoveLobesApart, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, 1, totalEpochs)

    learningLayers.attachLearningStrategy(EstimateInputDistribution, strategyArguments)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, epochsForEstimation=1, totalEpochs)

    return mse, totalEpochs
  end
end

module SelfOrgMixtureWithContext

  def selfOrgWithContextStrategyArgs
    {
        :classOfInputDistributionModel => ExampleDistributionModel, :desiredMeanNetInput => 3.0,
        :ioFunction => SigmoidIOFunction, :extendStrategyWithModule => LearningSuppressionViaLink
    }
  end
end

#module SupervisedTrainingWithContext
#
#  def selfOrgWithContextStrategyArgs
#    {
#        :classOfInputDistributionModel => ExampleDistributionModel, :desiredMeanNetInput => 3.0,
#        :extendStrategyWithModule => LearningSuppressionViaLink
#    }
#  end
#end


###################################################################
; ######################## OneNeuronSelfOrgTrainer #############

class OneNeuronSelfOrgTrainer < TrainerBase
  include SelfOrgMixture

  def train
    distributeSetOfExamples(examples)

    totalEpochs = 0
    ioFunction = SigmoidIOFunction #NonMonotonicIOFunction

    ### self-org 1st hidden layer
    learningLayers = outputLayer
    mse, totalEpochs = selOrg(selfOrgNoContextStrategyArgs, learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    #display:
    logger.puts "Output Layer BEFORE SUPERVISED TRAINING:"
    forEachExampleDisplayInputsAndOutputs(outputLayer)

    #learningLayers = outputLayer.to_LayerAry
    #mse, totalEpochs = supervisedTraining(learningLayers, ioFunction, args[:epochsForSupervisedTraining], totalEpochs)
    #
    #logger.puts "Output Layer AFTER SUPERVISED TRAINING:"
    #forEachExampleDisplayInputsAndOutputs(outputLayer)

    return totalEpochs, mse, 0.0 # calcTestingMeanSquaredErrors
  end

  def supervisedTraining(learningLayers, ioFunction, epochsDuringPhase, totalEpochs)
    propagatingLayers, xxxxxx = layerDetermination(learningLayers)

    strategyArguments = {:ioFunction => ioFunction}
    learningLayers.attachLearningStrategy(LearningBP, strategyArguments) if learningLayers.include?(outputLayer)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, epochsDuringPhase, totalEpochs)
    return mse, totalEpochs
  end

end

###################################################################
; ######################## MixtureTrainer3SelfOrgContextSuper # TODO ???? !!!! #############

class MixtureTrainer3SelfOrgContextSuper < TrainerBase
  attr_accessor :hiddenLayer1, :hiddenLayer2
  include SelfOrgMixture
  include SelfOrgMixtureWithContext
  # include SupervisedTraining

  def postInitialize
    @hiddenLayer1 = allNeuronLayers[1]
    @hiddenLayer2 = allNeuronLayers[2]
  end

  def train
    distributeSetOfExamples(examples)

    totalEpochs = 0
    ioFunction = SigmoidIOFunction

    ### self-org 1st hidden layer
    learningLayers = hiddenLayer1
    mse, totalEpochs = selOrg(selfOrgNoContextStrategyArgs, learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)
    puts "FIRST LAYER PROCESSING DONE"

    ## self-org 2nd hidden layer WITH CONTEXT!!
    learningLayers = hiddenLayer2

    # mse, totalEpochs = selOrg(selfOrgWithContextStrategyArgs, learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)
    mse, totalEpochs = selOrg(selfOrgNoContextStrategyArgs, learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)
    puts "SECOND LAYER PROCESSING DONE"

    mse, totalEpochs = temporaryHookName(ioFunction, mse, totalEpochs)

    logger.puts "Hidden Layer 2 outputs:"
    forEachExampleDisplayInputsAndOutputs(hiddenLayer2)

    learningLayers = outputLayer
    mse, totalEpochs = supervisedTraining(strategyArgs={}, learningLayers, ioFunction, args[:epochsForSupervisedTraining], totalEpochs)

    logger.puts "Output Layer AFTER SUPERVISED TRAINING:"
    forEachExampleDisplayInputsAndOutputs(outputLayer)

    return totalEpochs, mse, 0.0 # calcTestingMeanSquaredErrors
  end

  def supervisedTraining(strategyArgs, learningLayers, ioFunction, epochsDuringPhase, totalEpochs)
    propagatingLayers, notUsed = layerDetermination(learningLayers)
    strategyArgs[:ioFunction] = ioFunction

    learningLayers.attachLearningStrategy(LearningBP, strategyArgs)
    mse, totalEpochs = trainingPhaseFor(propagatingLayers, learningLayers, epochsDuringPhase, totalEpochs)
    return mse, totalEpochs
  end

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
  end

  def temporaryHookName(ioFunction, mse, totalEpochs)
    return [mse, totalEpochs]
  end
end

###################################################################
###################################################################
; ######################## Trainer4SelfOrgContextSuper #############

class Trainer4SelfOrgContextSuper < MixtureTrainer3SelfOrgContextSuper

  def temporaryHookName(ioFunction, mse, totalEpochs)
    selfOrgWithOutContext2ndLayer(ioFunction, mse, totalEpochs)
  end

  def selfOrgWithOutContext2ndLayer(selfOrgNoContextStrategyArgs, ioFunction, mse, totalEpochs)
    ### self-org 2nd hidden layer withOUT context!!
    learningLayers = hiddenLayer2
    mse, totalEpochs = selOrg(selfOrgNoContextStrategyArgs, learningLayers, ioFunction, args[:epochsForSelfOrg], totalEpochs)

    # Putting Hidden Layer 2 "back into context" with Outputs in Context (i.e., with 'dont know' representation added back)
    layerReceivingContext = hiddenLayer2
    forwardPropWithContext(layerReceivingContext, ioFunction)

    return [mse, totalEpochs]
  end
end


class TrainerAD1 < MixtureTrainer3SelfOrgContextSuper
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


