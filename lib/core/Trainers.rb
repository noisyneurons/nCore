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
                :bpFlockErrorAdaptingNeurons, :bpFlockErrorAdaptingLayers,
                :bpFlockErrorGeneratingNeurons, :bpFlockErrorGeneratingLayers,

                :maxFlockingIterationsCount, :flockingLearningRate, :bpFlockingLearningRate,
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
    @layersWhoseClustersNeedToBeSeeded = neuronGroups.layersWhoseClustersNeedToBeSeeded
    @neuronsWhoseClustersNeedToBeSeeded = neuronGroups.neuronsWhoseClustersNeedToBeSeeded

    @outputErrorAdaptingLayers = neuronGroups.outputErrorAdaptingLayers
    @flockErrorGeneratingLayers = neuronGroups.flockErrorGeneratingLayers
    @flockErrorAdaptingLayers = neuronGroups.flockErrorAdaptingLayers
    @bpFlockErrorAdaptingLayers = neuronGroups.bpFlockErrorAdaptingLayers
    @bpFlockErrorGeneratingLayers = neuronGroups.bpFlockErrorGeneratingLayers

    @outputErrorAdaptingNeurons = neuronGroups.outputErrorAdaptingNeurons
    @flockErrorGeneratingNeurons = neuronGroups.flockErrorGeneratingNeurons
    @flockErrorAdaptingNeurons = neuronGroups.flockErrorAdaptingNeurons
    @bpFlockErrorAdaptingNeurons = neuronGroups.bpFlockErrorAdaptingNeurons
    @bpFlockErrorGeneratingNeurons = neuronGroups.bpFlockErrorGeneratingNeurons
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
    testMSE = calcTestingMeanSquaredErrors
    return calcMSE, testMSE, accumulatedAbsoluteFlockingErrors
  end

  def performStandardBackPropTraining
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }
    acrossExamplesAccumulateDeltaWs(outputErrorAdaptingNeurons) { |aNeuron, dataRecord| aNeuron.calcAccumDeltaWsForHigherLayerError }
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

class StepTrainCircleProblemLocFlockAtOutputNeuron < AbstractStepTrainer

  def train(trials)
    distributeSetOfExamples(examples)
    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded) # TODO ONLY "seed" when necessary?   # unless(args[:epochs] > 1)
    self.accumulatedAbsoluteFlockingErrors = []
    mseMaxAllowedAfterFlocking = nil

    while (trainingSequence.stillMoreEpochs)
      initialMSEatBeginningOfBPOELoop = loopForBackPropOfOutputError()
      loopForLocalFlocking(initialMSEatBeginningOfBPOELoop)
    end
    testMSE = calcTestingMeanSquaredErrors
    return calcMSE, testMSE, accumulatedAbsoluteFlockingErrors
  end

  def loopForBackPropOfOutputError
    saveInitialMSE = true
    adaptToOutputError = true
    while (adaptToOutputError)
      mseBeforeBackProp, mseAfterBackProp = performStandardBackPropTrainingWithExtraMeasures()
      initialMSEatBeginningOfBPOELoop = mseBeforeBackProp if (saveInitialMSE)
      saveInitialMSE = false
      adaptToOutputError = (initialMSEatBeginningOfBPOELoop * args[:ratioDropInMSE]) < mseAfterBackProp
      recordAndIncrementEpochs
    end
    initialMSEatBeginningOfBPOELoop
  end

  def loopForLocalFlocking(initialMSEatBeginningOfBPOELoop)
    mseMaxAllowedAfterLocalFlocking = initialMSEatBeginningOfBPOELoop * args[:ratioDropInMSEForFlocking]
    maxFlockingIterationsCount = args[:maxFlockingIterationsCount]
    targetFlockIterationsCount = args[:targetFlockIterationsCount]
    # zeroOutFlockingLinksMomentumMemoryStore
    flockCount = 0
    until ((flockCount += 1) > maxFlockingIterationsCount)
      zeroOutFlockingLinksMomentumMemoryStore
      recenterEachNeuronsClusters(flockErrorGeneratingNeurons)
      mseBeforeFlocking, mseAfterFlocking = adaptToLocalFlockErrorWithExtraMeasures()
      recordAndIncrementEpochs
      break if (mseAfterFlocking > mseMaxAllowedAfterLocalFlocking)
    end

    if (maxFlockingIterationsCount > 0)
      self.flockingLearningRate = flockingLearningRate * 0.707 if (flockCount < targetFlockIterationsCount)
      self.flockingLearningRate = flockingLearningRate * 1.414 if (flockCount > targetFlockIterationsCount)
      puts "flockCount=\t#{flockCount}\tflockLearningRate=\t#{flockingLearningRate}"
    end
  end

  def recordAndIncrementEpochs
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.dbStoreNeuronData }
    dbStoreTrainingData()
    trainingSequence.nextEpoch
  end

  def performStandardBackPropTrainingWithExtraMeasures
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }
    acrossExamplesAccumulateDeltaWs(outputErrorAdaptingNeurons) { |aNeuron, dataRecord| aNeuron.calcAccumDeltaWsForHigherLayerError }
    mseBeforeBackProp = calcMSE # assumes squared error for each example and output neuron is stored in NeuronRecorder
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
    mseAfterBackProp = calcMeanSumSquaredErrors # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder
    return mseBeforeBackProp, mseAfterBackProp
  end

  def adaptToLocalFlockErrorWithExtraMeasures
    STDERR.puts "Generating neurons and adapting neurons are not one in the same.  This is NOT local flocking!!" if (flockErrorGeneratingNeurons != flockErrorAdaptingNeurons)
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = flockingLearningRate }
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError = 0.0 } # accumulatedAbsoluteFlockingError is a metric used for global control and monitoring
    acrossExamplesAccumulateFlockingErrorDeltaWs
    mseBeforeFlocking = calcMSE # assumes squared error for each example and output neuron is stored in NeuronRecorder
    self.accumulatedAbsoluteFlockingErrors = calcAccumulatedAbsoluteFlockingErrors(flockErrorGeneratingNeurons)
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
    mseAfterFlocking = calcMeanSumSquaredErrors # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder
                                                                                                  # std("flocking\t", [mseBeforeFlocking, mseAfterFlocking])
    return mseBeforeFlocking, mseAfterFlocking
  end

end

class StepTrainCircleProblemBPFlockAndLocFlockAtOutputNeuron < StepTrainCircleProblemLocFlockAtOutputNeuron
  attr_accessor :bpFlockingLearningRate

  def train(trials)
    distributeSetOfExamples(examples)
    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded) # TODO ONLY "seed" when necessary?   # unless(args[:epochs] > 1)
    self.accumulatedAbsoluteFlockingErrors = []

    while (trainingSequence.stillMoreEpochs)

      initialMSEatBeginningOfBPOELoop = loopForBackPropOfOutputError()
      loopForLocalFlocking(initialMSEatBeginningOfBPOELoop)
      loopForBackPropOfFlockingError(initialMSEatBeginningOfBPOELoop)

    end
    testMSE = calcTestingMeanSquaredErrors
    return calcMSE, testMSE, accumulatedAbsoluteFlockingErrors
  end

  def loopForBackPropOfFlockingError(initialMSEatBeginningOfBPOELoop)
    mseMaxAllowedAfterBPFlocking = initialMSEatBeginningOfBPOELoop * args[:ratioDropInMSEForBPFlocking]
    maxBPFlockingIterationsCount = args[:maxBPFlockingIterationsCount]
    targetBPFlockIterationsCount = args[:targetBPFlockIterationsCount]

    flockCount = 0
    until ((flockCount += 1) > maxBPFlockingIterationsCount)
      recenterEachNeuronsClusters(bpFlockErrorGeneratingNeurons)
      mseBeforeFlocking, mseAfterFlocking = adaptToBPofFlockErrorWithExtraMeasures()
      recordAndIncrementEpochs
      break if (mseAfterFlocking > mseMaxAllowedAfterBPFlocking)
    end

    if (maxBPFlockingIterationsCount > 0)
      self.bpFlockingLearningRate = bpFlockingLearningRate * 0.707 if (flockCount < targetBPFlockIterationsCount)
      self.bpFlockingLearningRate = bpFlockingLearningRate * 1.414 if (flockCount > targetBPFlockIterationsCount)
      puts "bpFlockCount=\t#{flockCount}\tbpFlockingLearningRate\t#{bpFlockingLearningRate}"
    end
  end

  def adaptToBPofFlockErrorWithExtraMeasures
    bpFlockErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = bpFlockingLearningRate }
    bpFlockErrorGeneratingNeurons.each { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError = 0.0 }
    acrossExamplesAccumulateBackPropedFlockingErrorDeltaWs
    mseBeforeFlocking = calcMSE # assumes squared error for each example and output neuron is stored in NeuronRecorder
    self.accumulatedAbsoluteFlockingErrors = calcAccumulatedAbsoluteFlockingErrors(bpFlockErrorGeneratingNeurons)
    bpFlockErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
    mseAfterFlocking = calcMeanSumSquaredErrors # Does NOT assume squared error for each example and output neuron is stored in NeuronRecorder                                                                                                 # std("flocking\t", [mseBeforeFlocking, mseAfterFlocking])
    return mseBeforeFlocking, mseAfterFlocking
  end

  def acrossExamplesAccumulateBackPropedFlockingErrorDeltaWs
    acrossExamplesAccumulateDeltaWs(bpFlockErrorGeneratingNeurons) do |aNeuron, dataRecord|
      dataRecord[:localFlockingError] = calcNeuronsLocalFlockingError(aNeuron)
      aNeuron.backPropagate { |higherError, localFlockingError| localFlockingError }
      bpFlockErrorAdaptingNeurons.each do |anAdaptingNeuron|
        anAdaptingNeuron.backPropagate
        anAdaptingNeuron.calcAccumDeltaWsForHigherLayerError
      end
    end
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
      mse, testMSE, accumulatedAbsoluteFlockingErrors = stepTrainer.train(numTrials)
    end
    arrayOfNeuronsToPlot = [network.outputLayer[0]]
    plotTrainingResults(arrayOfNeuronsToPlot)
    return trainingSequence.epochs, mse, testMSE, accumulatedAbsoluteFlockingErrors
  end

  def plotTrainingResults(arrayOfNeuronsToPlot)
    generatePlotForEachNeuron(arrayOfNeuronsToPlot) if arrayOfNeuronsToPlot.present?
  end
end

class TrainingSuperONLYLocalFlocking < TrainingSupervisorBase
  def postInitialize
    self.neuronGroups = NeuronGroupsForSingleLayerNetwork.new(network)
    self.stepTrainer = StepTrainerForONLYLocalFlocking.new(examples, neuronGroups, trainingSequence, args)
  end

  def train
    mse = 1e20
    numTrials = 10
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      mse, accumulatedAbsoluteFlockingErrors = stepTrainer.train(numTrials)
    end
    arrayOfNeuronsToPlot = [network.outputLayer[0]]
    plotTrainingResults(arrayOfNeuronsToPlot)
    return trainingSequence.epochs, mse, accumulatedAbsoluteFlockingErrors
  end
end

class TrainingSupervisorOutputNeuronLocalFlocking < TrainingSupervisorBase
  def postInitialize
    self.neuronGroups = NeuronGroupsForSingleLayerNetwork.new(network)
    self.stepTrainer = StepTrainerForLocalFlockingAndOutputError.new(examples, neuronGroups, trainingSequence, args)
  end
end

class TrainingSupervisorHiddenNeuronLocalFlocking < TrainingSupervisorBase
  include RecordingAndPlottingRoutines

  def postInitialize
    self.neuronGroups = NeuronGroupsHiddenLayerLocalFlocking.new(network)
    self.stepTrainer = StepTrainerForLocalFlockingAndOutputError.new(examples, neuronGroups, trainingSequence, args)
  end
end

class TrainingSupervisorAllLocalFlockingLayers < TrainingSupervisorBase
  include RecordingAndPlottingRoutines

  def postInitialize
    self.neuronGroups = NeuronGroups3LayersAllLocalFlockingLayers.new(network)
    self.stepTrainer = StepTrainerForLocalFlockingAndOutputError2.new(examples, neuronGroups, trainingSequence, args)
  end
end


class TrainSuperCircleProblemLocFlockAtOutputNeuron < TrainingSupervisorBase
  include RecordingAndPlottingRoutines

  def postInitialize
    self.neuronGroups = NeuronGroups3LayersOutputLayerLocalFlocking.new(network)
    self.stepTrainer = StepTrainCircleProblemLocFlockAtOutputNeuron.new(examples, neuronGroups, trainingSequence, args)
  end
end


class TrainSuperCircleProblemBPFlockAndLocFlockAtOutputNeuron < TrainingSupervisorBase
  include RecordingAndPlottingRoutines

  def postInitialize
    self.neuronGroups = NeuronGroups3LayersOutputLocalAndBPFlocking.new(network)
    self.stepTrainer = StepTrainCircleProblemBPFlockAndLocFlockAtOutputNeuron.new(examples, neuronGroups, trainingSequence, args)
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

