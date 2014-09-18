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
  include DBAccess
  include RecordingAndPlottingRoutines

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
    phaseTrain { performStandardBackPropTraining }
    forEachExampleDisplayInputsAndOutputs
    testMSE = calcTestingMeanSquaredErrors
    return trainingSequence.epochs, calcMSE, testMSE
  end

  def phaseTrain
    mse = 1e100
    while ((mse >= minMSE) && trainingSequence.stillMoreEpochs)
      yield
      neuronsWithInputLinks.each { |aNeuron| aNeuron.dbStoreNeuronData }
      dbStoreTrainingData()
      trainingSequence.nextEpoch
      mse = calcMSE
    end
    trainingSequence.startNextPhaseOfTraining
    return mse
  end

  def performStandardBackPropTraining
    acrossExamplesAccumulateDeltaWs(neuronsWithInputLinks) { |aNeuron, dataRecord| aNeuron.calcDeltaWsAndAccumulate }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
  end

  def acrossExamplesAccumulateDeltaWs(neurons)
    clearEpochAccumulationsInAllNeurons()
    numberOfExamples.times do |exampleNumber|
      propagateAcrossEntireNetwork(exampleNumber)
      backpropagateAcrossEntireNetwork()
      calcWeightedErrorMetricForExample()
      neurons.each do |aNeuron|
        dataRecord = aNeuron.recordResponsesForExample
        yield(aNeuron, dataRecord, exampleNumber)
        aNeuron.dbStoreDetailedData
      end
    end
  end

  ###------------  Core Support Section ------------------------------
  #### Also, some refinements and specialized control functions:

  def calcMSE # assumes squared error for each example and output neuron is stored in NeuronRecorder
    sse = outputLayer.inject(0.0) { |sum, anOutputNeuron| sum + anOutputNeuron.calcSumOfSquaredErrors }
    return (sse / (numberOfOutputNeurons * numberOfExamples))
  end

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

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
  end

  def clearEpochAccumulationsInAllNeurons
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.error = 0.0 }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  end

  def propagateAcrossEntireNetwork(exampleNumber)
    allNeuronsInOneArray.flatten.each { |aNeuron| aNeuron.propagate(exampleNumber) }
  end

  def propagateExampleUpToLayer(exampleNumber, lastLayerOfNeuronsToReceivePropagation)
    allNeuronLayers.each do |aLayer|
      aLayer.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      break if lastLayerOfNeuronsToReceivePropagation == aLayer
    end
  end

  def clearStoredNetInputs
    allNeuronsInOneArray.each { |aNeuron| aNeuron.clearStoredNetInputs }
  end

  def storeNetInputsForExample
    allNeuronsInOneArray.each { |aNeuron| aNeuron.storeNetInputForExample }
  end

  def backpropagateAcrossEntireNetwork
    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
  end

  def calcWeightedErrorMetricForExample
    outputLayer.collect { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
  end
end


###################################################################


class Trainer7pt1 < TrainerBase
#  include SelfOrgTraining

  def train
    distributeSetOfExamples(examples)

    #puts "phase1: short bp for hidden layer 1 + output layer"
    learningNeurons = allNeuronLayers[1] + outputLayer
    phaseTrain { performStandardBackPropTrainingOn(learningNeurons) }

    puts "phase2: self-org for hidden layer 1 "
    phaseTrain { performSelfOrgTrainingOn(allNeuronLayers[1]) }

    #puts "phase3: short bp for hidden layer 1 + output layer"
    #phaseTrain { performStandardBackPropTrainingOn(learningNeurons) }

    #puts "phase4: context learning for hidden layer 2, controlled by hidden layer 1 "
    #phaseTrain { performStandardBackPropTrainingOn(neuronsWithInputLinks) }

    #puts "phase5: context learning for hidden layer 2, controlled by hidden layer 1"
    #phaseTrain { performSelfOrgTrainingOn( allNeuronLayers[2] ) }
    #
    #puts "phase6"
    #phaseTrain { performStandardBackPropTrainingOn(neuronsWithInputLinks) }
    #
    #allNeuronLayers[2].each do |aNeuron|
    #  aNeuron.neuronControllingLearning = theBiasNeuron
    #  aNeuron.reverseLearningProbability = false
    #end
    #
    #puts "phase7"
    #phaseTrain { performSelfOrgTrainingOn( allNeuronLayers[2] ) }
    #
    #puts "phase8"
    #phaseTrain { performStandardBackPropTrainingOn(neuronsWithInputLinks) }


    ##phase6 NON-context bp learning for entire network
    #trainingSequence.startNextPhaseOfTraining
    #phaseTrain { performStandardBackPropTrainingOn(neuronsWithInputLinks) }


    forEachExampleDisplayInputsAndOutputs
    testMSE = calcTestingMeanSquaredErrors
    return trainingSequence.epochs, calcMSE, testMSE
  end

  def performStandardBackPropTrainingOn(learningNeurons)
    zeroWeightsOfNon(learningNeurons)
    acrossExamplesAccumulateDeltaWs(learningNeurons) { |aNeuron, dataRecord| aNeuron.calcDeltaWsAndAccumulate }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
  end

  def zeroWeightsOfNon(learningNeurons)
    nonLearningNeurons = neuronsWithInputLinks - learningNeurons.flatten
    nonLearningNeurons.each { |neuron| neuron.zeroWeights }
  end
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

