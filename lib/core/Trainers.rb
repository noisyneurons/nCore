### VERSION "nCore"
## ../nCore/lib/core/Trainers.rb


######
class AbstractStepTrainer
  attr_accessor :examples, :numberOfExamples, :neuronGroups, :trainingSequence,
                :args, :numberOfOutputNeurons, :minMSE,

                :allNeuronLayers, :inputLayer, :outputLayer,
                :layersWithInputLinks, :layersWhoseClustersNeedToBeSeeded,

                :allNeuronsInOneArray, :neuronsWithInputLinks,
                :neuronsWithInputLinksInReverseOrder, :neuronsWhoseClustersNeedToBeSeeded,

                :outputErrorAdaptingLayers, :flockErrorGeneratingLayers, :flockErrorAdaptingLayers,
                :outputErrorAdaptingNeurons, :flockErrorGeneratingNeurons, :flockErrorAdaptingNeurons,
                :bpFlockErrorAdaptingNeurons, :bpFlockErrorAdaptingLayers,
                :bpFlockErrorGeneratingNeurons, :bpFlockErrorGeneratingLayers,
                :outputLayerNeurons, :hiddenLayerNeurons,

                :maxFlockingIterationsCount, :flockingLearningRate, :bpFlockingLearningRate,
                :flockingIterationsCount, # :accumulatedAbsoluteFlockingErrors,
                :accumulatedExampleImportanceFactors, :absFlockingErrorsOld

  include NeuronToNeuronConnection
  include ExampleDistribution
  include DBAccess

  def initialize(examples, neuronGroups, trainingSequence, args)
    @examples = examples
    @neuronGroups = neuronGroups
    @trainingSequence = trainingSequence
    @args = args

    specifyGroupsOfLayersAndNeurons()

    @numberOfOutputNeurons = @outputLayer.length
    @numberOfExamples = args[:numberOfExamples]
    @minMSE = args[:minMSE]
  end

  def specifyGroupsOfLayersAndNeurons
    @allNeuronLayers = neuronGroups.allNeuronLayers
    @allNeuronsInOneArray = neuronGroups.allNeuronsInOneArray
    @inputLayer = neuronGroups.inputLayer
    @outputLayer = neuronGroups.outputLayer
    @layersWithInputLinks = neuronGroups.layersWithInputLinks
    @neuronsWithInputLinks = neuronGroups.neuronsWithInputLinks
    @neuronsWithInputLinksInReverseOrder = neuronGroups.neuronsWithInputLinksInReverseOrder


    @outputLayerNeurons = neuronGroups.outputLayerNeurons
    @hiddenLayerNeurons = neuronGroups.hiddenLayerNeurons

    @outputErrorAdaptingLayers = neuronGroups.outputErrorAdaptingLayers
    @outputErrorAdaptingNeurons = neuronGroups.outputErrorAdaptingNeurons
  end

  def train
    distributeSetOfExamples(examples)
    mse = 1e100
    while ((mse >= minMSE) && trainingSequence.stillMoreEpochs)
      innerTrainingLoop()
      dbStoreTrainingData()
      trainingSequence.nextEpoch
      mse = calcMSE
     # std("mse 1= ", mse)
    end
    testMSE = calcTestingMeanSquaredErrors
    return trainingSequence.epochs, calcMSE, testMSE
  end

  def performStandardBackPropTraining
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }
    acrossExamplesAccumulateDeltaWs(outputErrorAdaptingNeurons) { |aNeuron, dataRecord| aNeuron.calcDeltaWsAndAccumulate }
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
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
      neuronsWithInputLinks.each { |aNeuron| aNeuron.learning = false }
      testMSE = genericCalcMeanSumSquaredErrors(numberOfTestingExamples)
      neuronsWithInputLinks.each { |aNeuron| aNeuron.learning = true }
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

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
  end

  def clearEpochAccumulationsInAllNeurons
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  end

  def propagateAcrossEntireNetwork(exampleNumber)
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
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

  # BEGIN: CODE FOR AUTOMATIC CONTROL OF LEARNING RATES PER LAYER
  # todo Code for automatic control of learning rates for bp of OE and FE.  Separate gain control per neuron necessary for local flocking. #############

  def performNormalizedBackPropTrainingWithExtraMeasures
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }
    acrossExamplesAccumulateDeltaWs(outputErrorAdaptingNeurons) { |aNeuron, dataRecord| aNeuron.calcAccumDeltaWsForHigherLayerError }
    mseBeforeBackProp = calcMSE # assumes squared error for each example and output neuron is stored in NeuronRecorder
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
    mseAfterBackProp, netInputs = calcMeanSumSquaredErrorsAndNetInputs # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder


    maxChangesInExampleGainsInEachLayer
    return mseBeforeBackProp, mseAfterBackProp
  end

  def measureChangesInExampleGainsInEachLayer

    maxLogGainChangeForEachLayer = outputErrorAdaptingLayers.collect do |aLayer|

      maxGainRatioForLayer = nil
      aLayer.each do |aNeuron|
        mostRecentGains, previousGains = recentAndPastExampleGains(aNeuron)
        maxGainRatioForLayer = calculateMaxGainRatio(mostRecentGains, previousGains)
      end
      maxGainRatioForLayer
    end


  end

  def calculateMaxGainRatio(mostRecentGains, previousGains)
    gainRatios = []
    previousGains.each_with_index do |previousGain, index|
      mostRecentGain = mostRecentGains[index]
      gainRatios << (previousGain / mostRecentGain) if (previousGain >= mostRecentGain)
      gainRatios << (mostRecentGain / previousGain) if (previousGain < mostRecentGain)
    end
    maxGainRatioForLayer = gainRatios.max
  end

  def recentAndPastExampleGains(aNeuron)
    withinEpochMeasures = aNeuron.metricRecorder.withinEpochMeasures

    previousGains = withinEpochMeasures.collect do |measuresForAnExample|
      aNeuron.ioDerivativeFromNetInput(measuresForAnExample[:netInput])
    end
    mostRecentGains = aNeuron.storedNetInputs.collect do |aNetInput|
      aNeuron.ioDerivativeFromNetInput(aNetInput)
    end
    return mostRecentGains, previousGains
  end

  # END CODE FOR AUTOMATIC CONTROL OF LEARNING RATES PER LAYER

end


class StepTrainerForOutputErrorBPOnly < AbstractStepTrainer
  def innerTrainingLoop
    performStandardBackPropTraining()
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.dbStoreNeuronData }
  end
end

class Step1TrainerForJumpLinksOutputErrorBPOnly < StepTrainerForOutputErrorBPOnly
  def performStandardBackPropTraining
    neuronsWithInputLinks.each { |aNeuron| aNeuron.learningRate = 0.0 }
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }

    zeroLearningRateInLinksBetweenNeurons(allNeuronLayers[1], outputErrorAdaptingNeurons)
    zeroWeightsInLinksBetweenNeurons(allNeuronLayers[1], outputErrorAdaptingNeurons)

    acrossExamplesAccumulateDeltaWs(outputErrorAdaptingNeurons) { |aNeuron, dataRecord| aNeuron.calcDeltaWsAndAccumulate }
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
  end
end

class Step2TrainerForJumpLinksOutputErrorBPOnly < StepTrainerForOutputErrorBPOnly
end



class StepTrainerForOutputErrorBPOnlyModLR < StepTrainerForOutputErrorBPOnly
  def performStandardBackPropTraining
    outputLayerNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputLayerLearningRate] }

    ###
    #hiddenLayerNeurons[0].learningRate = args[:hiddenLayerLearningRate]
    #hiddenLayerNeurons[1].learningRate = args[:hiddenLayerLearningRate] / 10.0
    #hiddenLayerNeurons[2].learningRate = args[:hiddenLayerLearningRate] / 100.0

    hiddenLayerNeurons.each { |aNeuron| aNeuron.learningRate = args[:hiddenLayerLearningRate] }

    ###

    acrossExamplesAccumulateDeltaWs(outputErrorAdaptingNeurons) { |aNeuron, dataRecord| aNeuron.calcDeltaWsAndAccumulate }
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
  end
end


###################################################################
###################################################################

class TrainingSupervisorBase
  attr_accessor :examples, :network, :args, :neuronGroups, :stepTrainer, :trainingSequence, :startTime, :elapsedTime, :minMSE
  include RecordingAndPlottingRoutines

  def initialize(examples, network, args)
    @examples = examples
    @network = network
    @args = args
    @trainingSequence = args[:trainingSequence]
    @startTime = Time.now
    @elapsedTime = nil
    @minMSE = args[:minMSE]
    postInitialize
  end

  def postInitialize
    STDERR.puts "postInitialize of base class called!"
  end

  def train
    stepTrainer.train
  end
end


class ThreeClass2HiddenSupervisorOEBP < TrainingSupervisorBase
  def postInitialize
    self.neuronGroups = GroupsForThreeClass2HiddenLayersOEBP.new(network)
    self.stepTrainer = StepTrainerForOutputErrorBPOnly.new(examples, neuronGroups, trainingSequence, args)
  end
end


class StandardBPTrainingSupervisor < TrainingSupervisorBase
  def postInitialize
    self.neuronGroups = NeuronGroupsFor3LayerBPNetwork.new(network)
    self.stepTrainer = StepTrainerForOutputErrorBPOnly.new(examples, neuronGroups, trainingSequence, args)
  end
end


class Project6pt2TrainingSupervisor < TrainingSupervisorBase
  attr_accessor :stepTrainer1, :stepTrainer2, :neuronGroups1, :neuronGroups2

  def postInitialize
    self.neuronGroups1 = NeuronGroupsForStep1JumpLinked3LayerNetwork.new(network)
    self.stepTrainer1 = Step1TrainerForJumpLinksOutputErrorBPOnly.new(examples, neuronGroups1, trainingSequence, args)

    self.neuronGroups2 = NeuronGroupsForStep2JumpLinked3LayerNetwork.new(network)
    trainingSequence = TrainingSequence.new(args)
    trainingSequence.maxNumberOfEpochs = 7e3
    self.stepTrainer2 = Step2TrainerForJumpLinksOutputErrorBPOnly.new(examples, neuronGroups2, trainingSequence, args)
  end

  def train
    stepTrainer1.train
    puts network
    stepTrainer2.train
  end

end


class StandardBPTrainingSupervisorModLR < StandardBPTrainingSupervisor
  def postInitialize
    self.neuronGroups = NeuronGroupsFor3LayerBPNetworkModLR.new(network)
    self.stepTrainer = StepTrainerForOutputErrorBPOnlyModLR.new(examples, neuronGroups, trainingSequence, args)
  end
end


class BPTrainingSupervisorFor1LayerNet < TrainingSupervisorBase
  def postInitialize
    self.neuronGroups = NeuronGroupsFor1LayerBPNetwork.new(network)
    self.stepTrainer = StepTrainerForOutputErrorBPOnly.new(examples, neuronGroups, trainingSequence, args)
  end
end


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
#

