### VERSION "nCore"
## ../nCore/lib/core/Trainers.rb


###########...

######
module FlockingDecisionRoutines # TODO If one neuron does NOT meet criteria, all flocking neurons to continue to flock. Why not just those neurons that have not met the criteria.

  def flockingShouldOccur?(accumulatedAbsoluteFlockingErrors)
    return doNotFlock() if (tooEarlyToFlock?)
    if (doWeNeedToFlockNOW?(accumulatedAbsoluteFlockingErrors))
      return yesDoFlock()
    else
      return doNotFlock()
    end
  end

  def doWeNeedToFlockNOW?(accumulatedAbsoluteFlockingErrors)
    return true if (accumulatedAbsoluteFlockingErrors.empty?)
    needToReduceFlockingError = determineIfWeNeedToReduceFlockingError
    stillEnoughIterationsToFlock = flockingIterationsCount < maxFlockingIterationsCount
    return (needToReduceFlockingError && stillEnoughIterationsToFlock)
  end

  def determineIfWeNeedToReduceFlockingError
    largestAbsoluteFlockingErrorPerExample = accumulatedAbsoluteFlockingErrors.max / numberOfExamples
    return (largestAbsoluteFlockingErrorPerExample > args[:maxAbsFlockingErrorsPerExample])
  end

  def yesDoFlock
    self.flockingIterationsCount += 1
    true
  end

  def doNotFlock
    self.absFlockingErrorsOld = []
    self.flockingIterationsCount = 0
    return false
  end

  def tooEarlyToFlock?
    args[:epochs] < args[:epochsBeforeFlockingAllowed] #  200
  end
end

class AbstractStepTrainer
  attr_accessor :examples, :numberOfExamples, :neuronGroups, :trainingSequence,
                :args, :numberOfOutputNeurons,

                :allNeuronLayers, :inputLayer, :outputLayer,
                :layersWithInputLinks, :layersWhoseClustersNeedToBeSeeded,

                :allNeuronsInOneArray, :neuronsWithInputLinks,
                :neuronsWithInputLinksInReverseOrder, :neuronsWhoseClustersNeedToBeSeeded,

                :outputErrorAdaptingLayers, :flockErrorGeneratingLayers, :flockErrorAdaptingLayers,
                :outputErrorAdaptingNeurons, :flockErrorGeneratingNeurons, :flockErrorAdaptingNeurons,

                :maxFlockingIterationsCount, :flockingLearningRate,
                :flockingIterationsCount, :accumulatedAbsoluteFlockingErrors,
                :accumulatedExampleImportanceFactors, :absFlockingErrorsOld

  include ExampleDistribution
  include FlockingDecisionRoutines
  include DBAccess

  def initialize(examples, neuronGroups, trainingSequence, args)
    @examples = examples
    @neuronGroups = neuronGroups
    @trainingSequence = trainingSequence
    @args = args

    specifyGroupsOfLayersAndNeurons()

    @numberOfOutputNeurons = @outputLayer.length
    @numberOfExamples = args[:numberOfExamples]

    @flockingIterationsCount = 0
    @accumulatedExampleImportanceFactors = nil
    @flockingLearningRate = args[:flockingLearningRate]
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
    @layersWhoseClustersNeedToBeSeeded = neuronGroups.layersWhoseClustersNeedToBeSeeded
    @neuronsWhoseClustersNeedToBeSeeded = neuronGroups.neuronsWhoseClustersNeedToBeSeeded

    @outputErrorAdaptingLayers = neuronGroups.outputErrorAdaptingLayers
    @flockErrorGeneratingLayers = neuronGroups.flockErrorGeneratingLayers
    @flockErrorAdaptingLayers = neuronGroups.flockErrorAdaptingLayers
    @outputErrorAdaptingNeurons = neuronGroups.outputErrorAdaptingNeurons
    @flockErrorGeneratingNeurons = neuronGroups.flockErrorGeneratingNeurons
    @flockErrorAdaptingNeurons = neuronGroups.flockErrorAdaptingNeurons
  end

  def train(trials)
    distributeSetOfExamples(examples)
    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded) # TODO ONLY "seed" when necessary?   # unless(args[:epochs] > 1)
    self.accumulatedAbsoluteFlockingErrors = []
    trials.times do
      innerTrainingLoop()
      dbStoreTrainingData()
      trainingSequence.nextEpoch
    end
    return calcMSE, accumulatedAbsoluteFlockingErrors
  end

  def performStandardBackPropTraining
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }
    acrossExamplesAccumulateDeltaWs(outputErrorAdaptingNeurons) { |aNeuron, dataRecord| aNeuron.calcAccumDeltaWsForHigherLayerError }
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
  end

  def performStandardBackPropTrainingWithExtraMeasures
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }
    acrossExamplesAccumulateDeltaWs(outputErrorAdaptingNeurons) { |aNeuron, dataRecord| aNeuron.calcAccumDeltaWsForHigherLayerError }
    mseBeforeBackProp = calcMSE # assumes squared error for each example and output neuron is stored in NeuronRecorder
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
    mseAfterBackProp = calcMeanSumSquaredErrors # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder
    return mseBeforeBackProp, mseAfterBackProp
  end

  def adaptToLocalFlockError
    STDERR.puts "Generating neurons and adapting neurons are not one in the same.  This is NOT local flocking!!" if (flockErrorGeneratingNeurons != flockErrorAdaptingNeurons)
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = flockingLearningRate }
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError = 0.0 } # accumulatedAbsoluteFlockingError is a metric used for global control and monitoring
    acrossExamplesAccumulateFlockingErrorDeltaWs
    self.accumulatedAbsoluteFlockingErrors = calcAccumulatedAbsoluteFlockingErrors()
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight } if (determineIfWeNeedToReduceFlockingError)
  end

  def adaptToLocalFlockErrorWithExtraMeasures
    STDERR.puts "Generating neurons and adapting neurons are not one in the same.  This is NOT local flocking!!" if (flockErrorGeneratingNeurons != flockErrorAdaptingNeurons)
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = flockingLearningRate }
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError = 0.0 } # accumulatedAbsoluteFlockingError is a metric used for global control and monitoring
    acrossExamplesAccumulateFlockingErrorDeltaWs
    mseBeforeFlocking = calcMSE # assumes squared error for each example and output neuron is stored in NeuronRecorder
    self.accumulatedAbsoluteFlockingErrors = calcAccumulatedAbsoluteFlockingErrors()
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
    mseAfterFlocking = calcMeanSumSquaredErrors # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder
                                                                                                  # std("flocking\t", [mseBeforeFlocking, mseAfterFlocking])
    return mseBeforeFlocking, mseAfterFlocking
  end

  def acrossExamplesAccumulateFlockingErrorDeltaWs
    acrossExamplesAccumulateDeltaWs(flockErrorAdaptingNeurons) do |aNeuron, dataRecord|
      dataRecord[:localFlockingError] = calcNeuronsLocalFlockingError(aNeuron)
      aNeuron.calcAccumDeltaWsForLocalFlocking
    end
  end

  def acrossExamplesAccumulateDeltaWs(adaptingNeurons)
    clearEpochAccumulationsInAllNeurons()
    numberOfExamples.times do |exampleNumber|
      propagateAcrossEntireNetwork(exampleNumber)
      backpropagateAcrossEntireNetwork()
      calcWeightedErrorMetricForExample()
      adaptingNeurons.each do |aNeuron|
        dataRecord = aNeuron.recordResponsesForExample
        yield(aNeuron, dataRecord, exampleNumber)
        aNeuron.dbStoreDetailedData
      end
    end
  end

  def calcNeuronsLocalFlockingError(aNeuron)
    localFlockingError = if (useFuzzyClusters?)
                           aNeuron.calcLocalFlockingError { aNeuron.targetForFlocking }
                         else
                           aNeuron.calcLocalFlockingError { aNeuron.centerOfDominantClusterForExample }
                         end
  end

  def calcAccumulatedAbsoluteFlockingErrors
    flockErrorGeneratingNeurons.collect { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError }
  end

  ###------------  Core Support Section ------------------------------
  #### Also, some refinements and specialized control functions:

  def calcMSE # assumes squared error for each example and output neuron is stored in NeuronRecorder
    sse = outputLayer.inject(0.0) { |sum, anOutputNeuron| sum + anOutputNeuron.calcSumOfSquaredErrors }
    return (sse / (numberOfOutputNeurons * numberOfExamples))
  end

  def calcMeanSumSquaredErrors # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder
    squaredErrors = []
    numberOfExamples.times do |exampleNumber|
      propagateAcrossEntireNetwork(exampleNumber)
      squaredErrors << calcWeightedErrorMetricForExample()
    end
    sse = squaredErrors.flatten.reduce(:+)
    return (sse / (numberOfExamples * numberOfOutputNeurons))
  end

  def calcTestingMeanSquaredErrors  # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder
    distributeSetOfExamples(args[:testingExamples])
    testMSE = calcMeanSumSquaredErrors
    distributeSetOfExamples(examples)
    return testMSE
  end

  def distributeSetOfExamples(examples)
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
  end

  def recenterEachNeuronsClusters(flockErrorGeneratingNeurons)
    dispersions = flockErrorGeneratingNeurons.collect { |aNeuron| aNeuron.clusterAllResponses } # TODO Perhaps we might only need to clusterAllResponses every K epochs?
  end

  def seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
      neuronsWithInputLinks.each { |aNeuron| aNeuron.recordResponsesForExample }
    end
    neuronsWhoseClustersNeedToBeSeeded.each { |aNeuron| aNeuron.initializeClusterCenters } # TODO is there any case where system should be reinitialized in this manner?
    recenterEachNeuronsClusters(neuronsWhoseClustersNeedToBeSeeded)
  end

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

  def backpropagateAcrossEntireNetwork
    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
  end

  def calcWeightedErrorMetricForExample
    outputLayer.collect { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
  end
end

class StepTrainerForONLYLocalFlocking < AbstractStepTrainer
  def innerTrainingLoop
    self.accumulatedAbsoluteFlockingErrors = []
    zeroOutFlockingLinksMomentumMemoryStore
    recenterEachNeuronsClusters(flockErrorGeneratingNeurons)
    adaptToLocalFlockError()
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.dbStoreNeuronData }
  end
end

class StepTrainerForOutputErrorBPOnly < AbstractStepTrainer
  def innerTrainingLoop
    performStandardBackPropTraining()
    self.accumulatedAbsoluteFlockingErrors = []
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.dbStoreNeuronData }
  end
end

class StepTrainerForLocalFlockingAndOutputError < AbstractStepTrainer
  def innerTrainingLoop
    unless (flockingShouldOccur?(accumulatedAbsoluteFlockingErrors))
      performStandardBackPropTraining()
      self.accumulatedAbsoluteFlockingErrors = []
    else
      zeroOutFlockingLinksMomentumMemoryStore
      recenterEachNeuronsClusters(flockErrorGeneratingNeurons)
      adaptToLocalFlockError()
    end
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.dbStoreNeuronData }
  end
end

class StepTrainerForLocalFlockingAndOutputError2 < AbstractStepTrainer

  def train(trials)
    distributeSetOfExamples(examples)
    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded) # TODO ONLY "seed" when necessary?   # unless(args[:epochs] > 1)
    maxFlockingIterationsCount = args[:maxFlockingIterationsCount]
    targetFlockIterationsCount = args[:targetFlockIterationsCount]
    self.accumulatedAbsoluteFlockingErrors = []
    mseMaxAllowedAfterFlocking = nil

    while (trainingSequence.stillMoreEpochs)

      saveInitialMSE = true
      adaptToOutputError = true
      while (adaptToOutputError)
        mseBeforeBackProp, mseAfterBackProp = performStandardBackPropTrainingWithExtraMeasures()
        initialMSEBeforeBackProp = mseBeforeBackProp if (saveInitialMSE)
        saveInitialMSE = false

        adaptToOutputError = (initialMSEBeforeBackProp * args[:ratioDropInMSE]) < mseAfterBackProp
        # puts "adaptToOutputError\t#{adaptToOutputError}\tinitialMSEBeforeBackProp\t#{initialMSEBeforeBackProp}\tmseAfterBackProp\t#{mseAfterBackProp}"
        mseMaxAllowedAfterFlocking = initialMSEBeforeBackProp * args[:ratioDropInMSEForFlocking]
        recordAndIncrementEpochs
      end

      flockCount = 0
      until ((flockCount += 1) > maxFlockingIterationsCount)
        zeroOutFlockingLinksMomentumMemoryStore
        recenterEachNeuronsClusters(flockErrorGeneratingNeurons)
        mseBeforeFlocking, mseAfterFlocking = adaptToLocalFlockErrorWithExtraMeasures()
        recordAndIncrementEpochs
        break if (mseAfterFlocking > mseMaxAllowedAfterFlocking)
      end

      self.flockingLearningRate = flockingLearningRate * 0.707 if (flockCount < targetFlockIterationsCount)
      self.flockingLearningRate = flockingLearningRate * 1.414 if (flockCount > targetFlockIterationsCount)
      puts "flockCount=\t#{flockCount}\tflockLearningRate=\t#{flockingLearningRate}"
    end
    return calcMSE, accumulatedAbsoluteFlockingErrors
  end

  def recordAndIncrementEpochs
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.dbStoreNeuronData }
    dbStoreTrainingData()
    trainingSequence.nextEpoch
  end
end


class StepTrainerForFlockingAndOutputError < AbstractStepTrainer
  def innerTrainingLoop
    unless (flockingShouldOccur?(accumulatedAbsoluteFlockingErrors))
      performStandardBackPropTraining()
      self.accumulatedAbsoluteFlockingErrors = []
    else
      zeroOutFlockingLinksMomentumMemoryStore
      recenterEachNeuronsClusters(flockErrorGeneratingNeurons)
      adaptToLocalFlockError()
    end
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.dbStoreNeuronData }
  end
end


######
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
    numTrials = 50
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      mse, accumulatedAbsoluteFlockingErrors = stepTrainer.train(numTrials)
    end
    arrayOfNeuronsToPlot = [network.outputLayer[0]]
    plotTrainingResults(arrayOfNeuronsToPlot)
    return trainingSequence.epochs, mse, accumulatedAbsoluteFlockingErrors
  end

  def plotTrainingResults(arrayOfNeuronsToPlot)
    generatePlotForEachNeuron(arrayOfNeuronsToPlot) if arrayOfNeuronsToPlot.present?
  end
end

class TrainingSuperONLYLocalFlocking < TrainingSupervisorBase
  def postInitialize
    self.neuronGroups = NeuronGroupsTrivial.new(network)
    self.stepTrainer = StepTrainerForONLYLocalFlocking.new(examples, neuronGroups, trainingSequence, args)
  end
end

class TrainingSupervisorOutputNeuronLocalFlocking < TrainingSupervisorBase
  def postInitialize
    self.neuronGroups = NeuronGroupsTrivial.new(network)
    self.stepTrainer = StepTrainerForLocalFlockingAndOutputError.new(examples, neuronGroups, trainingSequence, args)
  end
end

class TrainingSupervisorHiddenNeuronLocalFlocking < TrainingSupervisorBase
  include RecordingAndPlottingRoutines

  def postInitialize
    self.neuronGroups = NeuronGroupsHiddenNeuronLocalFlockingError.new(network)
    self.stepTrainer = StepTrainerForLocalFlockingAndOutputError.new(examples, neuronGroups, trainingSequence, args)
  end
end

class TrainingSupervisorAllLocalFlockingLayers < TrainingSupervisorBase
  include RecordingAndPlottingRoutines

  def postInitialize
    self.neuronGroups = NeuronGroupsAllLocalFlockingLayers.new(network)
    self.stepTrainer = StepTrainerForLocalFlockingAndOutputError2.new(examples, neuronGroups, trainingSequence, args)
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

