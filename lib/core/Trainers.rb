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
                :flockingIterationsCount, :accumulatedAbsoluteFlockingErrors,
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

    @flockingIterationsCount = 0
    @accumulatedExampleImportanceFactors = nil
    @flockingLearningRate = args[:flockingLearningRate]
    @bpFlockingLearningRate = args[:bpFlockingLearningRate]
    @maxFlockingIterationsCount = args[:maxFlockingIterationsCount]
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
    self.accumulatedAbsoluteFlockingErrors = []
    mse = 1e100
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      innerTrainingLoop()
      dbStoreTrainingData()
      trainingSequence.nextEpoch
      mse = calcMSE
    end
    testMSE = calcTestingMeanSquaredErrors
    return calcMSE, testMSE, accumulatedAbsoluteFlockingErrors
  end

  def performStandardBackPropTraining
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }
    acrossExamplesAccumulateDeltaWs(outputErrorAdaptingNeurons) { |aNeuron, dataRecord| aNeuron.calcDeltaWsAndAccumulate }
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
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

  def adaptToLocalFlockError
    STDERR.puts "Generating neurons and adapting neurons are not one in the same.  This is NOT local flocking!!" if (flockErrorGeneratingNeurons != flockErrorAdaptingNeurons)
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = flockingLearningRate }
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError = 0.0 } # accumulatedAbsoluteFlockingError is a metric used for global control and monitoring
    acrossExamplesAccumulateFlockingErrorDeltaWs
    self.accumulatedAbsoluteFlockingErrors = calcAccumulatedAbsoluteFlockingErrors(flockErrorGeneratingNeurons)
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight } #if (determineIfWeNeedToReduceFlockingError)
  end

  def acrossExamplesAccumulateFlockingErrorDeltaWs
    acrossExamplesAccumulateDeltaWs(flockErrorAdaptingNeurons) do |aNeuron, dataRecord|
      dataRecord[:localFlockingError] = calcNeuronsLocalFlockingError(aNeuron)
      aNeuron.calcAccumDeltaWsForLocalFlocking
    end
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

  def calcNeuronsLocalFlockingError(aNeuron)
    localFlockingError = aNeuron.calcLocalFlockingError
  end

  def calcAccumulatedAbsoluteFlockingErrors(flockErrorGeneratingNeurons)
    flockErrorGeneratingNeurons.collect { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError }
  end

  ###------------  Core Support Section ------------------------------
  #### Also, some refinements and specialized control functions:

  def calcMSE # assumes squared error for each example and output neuron is stored in NeuronRecorder
    sse = outputLayer.inject(0.0) { |sum, anOutputNeuron| sum + anOutputNeuron.calcSumOfSquaredErrors }
    return (sse / (numberOfOutputNeurons * numberOfExamples))
  end

  def calcMeanSumSquaredErrors # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder
    #    clearStoredNetInputs
    squaredErrors = []
    numberOfExamples.times do |exampleNumber|
      propagateAcrossEntireNetwork(exampleNumber)
      squaredErrors << calcWeightedErrorMetricForExample()
      #     storeNetInputsForExample
    end
    sse = squaredErrors.flatten.reduce(:+)
    return (sse / (numberOfExamples * numberOfOutputNeurons))
  end

  def calcTestingMeanSquaredErrors # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder
    testMSE = nil
    testingExamples = args[:testingExamples]
    unless (testingExamples.nil?)
      distributeSetOfExamples(testingExamples)
      testMSE = calcMeanSumSquaredErrors
      distributeSetOfExamples(examples)
    end
    return testMSE
  end

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
  end

  def recenterEachNeuronsClusters(arrayOfNeurons)
    dispersions = arrayOfNeurons.collect { |aNeuron| aNeuron.clusterAllResponses } # TODO Perhaps we might only need to clusterAllResponses every K epochs?
  end

  #def seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded)
  #  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  #  numberOfExamples.times do |exampleNumber|
  #    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
  #    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
  #    neuronsWithInputLinks.each { |aNeuron| aNeuron.recordResponsesForExample }
  #  end
  #  neuronsWhoseClustersNeedToBeSeeded.each { |aNeuron| aNeuron.initializeClusterCenters } # TODO is there any case where system should be reinitialized in this manner?
  #  recenterEachNeuronsClusters(neuronsWhoseClustersNeedToBeSeeded)
  #end

  def zeroOutFlockingLinksMomentumMemoryStore
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.inputLinks.each { |aLink| aLink.store = 0.0 } }
  end

  def useFuzzyClusters? # TODO Would this be better put 'into each neuron'
    return true if (args[:alwaysUseFuzzyClusters])
    exampleWeightings = flockErrorGeneratingNeurons.first.clusters[0].membershipWeightForEachExample # TODO Why is only the first neuron used for this determination?
    criteria0 = 0.2
    criteria1 = 1.0 - criteria0
    count = 0
    exampleWeightings.each { |aWeight| count += 1 if (aWeight <= criteria0) }
    exampleWeightings.each { |aWeight| count += 1 if (aWeight > criteria1) }
    # puts "count=\t #{count}"
    return true if (count < numberOfExamples)
    return false
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
end


class StepTrainerForOutputErrorBPOnly < AbstractStepTrainer
  def innerTrainingLoop
    performStandardBackPropTraining()
    self.accumulatedAbsoluteFlockingErrors = []
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.dbStoreNeuronData }
  end
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
    mse = 1e20
    testMSE = nil
    accumulatedAbsoluteFlockingErrors = nil
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      mse, testMSE, accumulatedAbsoluteFlockingErrors = stepTrainer.train
    end
    arrayOfNeuronsToPlot = [network.outputLayer[0]]
    plotTrainingResults(arrayOfNeuronsToPlot)
    return trainingSequence.epochs, mse, testMSE, accumulatedAbsoluteFlockingErrors
  end

  def plotTrainingResults(arrayOfNeuronsToPlot)
    generatePlotForEachNeuron(arrayOfNeuronsToPlot) if arrayOfNeuronsToPlot.present?
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

