### VERSION "nCore"
## ../nCore/lib/core/Trainers.rb

require 'rubygems'

require_relative 'Utilities'

#### TODO Redis and database stuff -- should be moved to more appropriate place
require 'relix'

class Experiment
  attr_reader :descriptionOfExperiment, :args

  $redis = Redis.new
  # $redis.flushdb
  $redis.setnx("experimentNumber", 0)
  @@number = $redis.get("experimentNumber")

  def Experiment.number
    @@number
  end

  #def Experiment.deleteTable
  #  $redis.del("experimentNumber")
  #end

  def initialize(experimentDescription)
    $redis.incr("experimentNumber")
    @descriptionOfExperiment = descriptionOfExperiment
  end

  def save
    $redis.save
  end
end
#### Redis and database stuff -- should be moved to more appropriate place

module RecordingAndPlottingRoutines

  def storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, dPrimes)
    self.elapsedTime = Time.now - startTime
    dataStoreManager.storeEndOfTrainingMeasures(lastEpoch, lastTrainingMSE, lastTestingMSE, dPrimes, elapsedTime)
  end

  def logNetworksResponses(neuronsCreatingFlockingError)
    neuronsCreatingFlockingError.each { |aNeuron| aNeuron.recordResponsesForEpoch } if (neuronsCreatingFlockingError.present?)
    mse = network.calcNetworksMeanSquareError
    network.recordResponses
    mse
  end

  def generatePlotForEachNeuron(arrayOfNeuronsToPlot)
    arrayOfNeuronsToPlot.each do |theNeuron|
      xAry, yAry = getZeroXingExampleSet(theNeuron)
      plotDotsWhereOutputGtPt5(xAry, yAry, theNeuron, trainingSequence.epochs)
    end
  end

  def oneForwardPassEpoch(testingExamples)
    trainingExamples = examples
    distributeSetOfExamples(testingExamples)
    testMSE = measureNeuralResponsesForTesting
    distributeSetOfExamples(trainingExamples) # restore training examples
    return testMSE
  end

  def getZeroXingExampleSet(theNeuron)
    trainingExamples = examples
    minX1 = -4.0
    maxX1 = 4.0
    numberOfTrials = 100
    increment = (maxX1 - minX1) / numberOfTrials
    x0Array = []
    x1Array = []
    numberOfTrials.times do |index|
      x1 = minX1 + index * increment
      x0 = findZeroCrossingFast(x1, theNeuron)
      next if (x0.nil?)
      x1Array << x1
      x0Array << x0
    end
    self.examples = trainingExamples
    [x0Array, x1Array]
  end

  def measureNeuralResponsesForTesting
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      neuronsWithInputLinks.each { |aNeuron| aNeuron.recordResponsesForExample }
    end
    mse = network.calcNetworksMeanSquareError
  end

  def findZeroCrossingFast(x1, theNeuron)
    minX0 = -2.0
    maxX0 = 2.0
    range = maxX0 - minX0

    initialOutput = ioFunctionFor2inputs(minX0, x1, theNeuron)
    initialOutputLarge = false
    initialOutputLarge = true if (initialOutput >= 0.5)

    wasThereACrossing = false
    20.times do |count|
      range = maxX0 - minX0
      bisectedX0 = minX0 + (range / 2.0)
      currentOutput = ioFunctionFor2inputs(bisectedX0, x1, theNeuron)
      currentOutputLarge = false
      currentOutputLarge = true if (currentOutput > 0.5)
      if (currentOutputLarge != initialOutputLarge)
        maxX0 = bisectedX0
        wasThereACrossing = true
      else
        minX0 = bisectedX0
      end
    end
    return nil if (wasThereACrossing == false)
    estimatedCrossingForX0 = (minX0 + maxX0) / 2.0
    return estimatedCrossingForX0
  end

  def ioFunctionFor2inputs(x0, x1, theNeuron) # TODO should this be moved to an analysis or plotting class?
    examples = [{:inputs => [x0, x1], :exampleNumber => 0}]
    distributeDataToAnArrayOfNeurons(inputLayer, examples)
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(0) }
    return theNeuron.output
  end
end

class AbstractTrainer
  attr_accessor :trainingSequence, :minMSE, :maxNumEpochs, :examples, :dataArray, :args,
                :numberOfExamples, :dataStoreManager,
                :lastEpoch, :lastTrainingMSE, :lastTestingMSE, :startTime, :elapsedTime,
                :dPrimes, :dPrimesOld,

                :network, :allNeuronLayers,
                :allNeuronsInOneArray, :inputLayer, :hiddenLayer, :outputLayer, :theBiasNeuron,
                :hiddenLayer1, :hiddenLayer2, :hiddenLayer3, :allHiddenLayers,
                :neuronsWithInputLinks, :neuronsWithInputLinksInReverseOrder,


                :neuronsAdaptingOnlyToBPOutputError, :neuronsToAdaptToOutputError,
                :neuronsCreatingFlockingError, :adaptingNeurons,
                :neuronsAdaptingToBackPropedFlockingError, :neuronsWhoseClustersNeedToBeSeeded,

                :layersWithInputLinks, :layersAdaptingOnlyToBPOutputError, :layersToAdaptToOutputError,
                :layersCreatingFlockingError, :layersAdaptingToLocalFlockingError,
                :layersAdaptingToBackPropedFlockingError, :layersWhoseClustersNeedToBeSeeded,

                :learningRateNoFlockPhase1, :learningRateLocalFlockPhase2, :learningRateBPOutputErrorPhase2,
                :learningRateNoFlockPhase1, :learningRateForBackPropedFlockingErrorPhase2, :learningRateFlockPhase2,

                :layerTuners, :flockingHasConverged, :adaptingLayers, :adaptingNeurons

  include ExampleDistribution
  include NeuronToNeuronConnection
  include RecordingAndPlottingRoutines

  def initialize(trainingSequence, network, args)
    @trainingSequence = trainingSequence
    @network = network
    @args = args
    @numberOfExamples = args[:numberOfExamples]
    @dataStoreManager = SimulationDataStoreManager.instance

    @allNeuronLayers = network.createSimpleLearningANN
    @allNeuronsInOneArray = network.allNeuronsInOneArray
    @inputLayer = network.inputLayer
    @outputLayer = network.outputLayer
    @theBiasNeuron = network.theBiasNeuron

    @startTime = Time.now
    @elapsedTime = nil
    @minMSE = args[:minMSE]
    postInitialize()
  end

###------------  Core Support Section ------------------------------

  def distributeSetOfExamples(examples)
    @examples = examples
    @numberOfExamples = examples.length
    distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
  end

  def recenterEachNeuronsClusters(adaptingNeurons)
    dispersions = adaptingNeurons.collect { |aNeuron| aNeuron.clusterAllResponses } # TODO Perhaps we might only need to clusterAllResponses every K epochs?
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
end

class SimpleAdjustableLearningRateTrainer < AbstractTrainer
  attr_accessor :maxFlockingIterationsCount, :flockingIterationsCount, :accumulatedExampleImportanceFactors,
                :absFlockingErrorsOld#,  :accumulatedAbsoluteFlockingErrors

  def postInitialize
    @flockingIterationsCount = 0
    @maxFlockingIterationsCount = args[:maxFlockingIterationsCount]
    @accumulatedExampleImportanceFactors = nil
  end

  def simpleLearningWithFlocking(examples)
    nameTrainingGroupsAndSetLearningRatesForStep1OfTraining
    mse, accumulatedAbsoluteFlockingErrors = stepLearning(examples)
    return trainingSequence.epochs, mse, accumulatedAbsoluteFlockingErrors
  end

  def displayTrainingResults(arrayOfNeuronsToPlot)
    puts network
    generatePlotForEachNeuron(arrayOfNeuronsToPlot) if arrayOfNeuronsToPlot.present?
  end

  def stepLearning(examples)
    accumulatedAbsoluteFlockingErrors = nil

    distributeSetOfExamples(examples)
    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded) # TODO  Except for the 'first' step, is this always needed?
    adaptingNeurons.each { |aNeuron| aNeuron.inputLinks.each { |aLink| aLink.store = 0.0 } }

    flockingHasConverged = true
    mse = Float::MAX

    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      mse, accumulatedAbsoluteFlockingErrors = oneEpochOfAdaptingNetworkWeights(flockingHasConverged)
      flockingHasConverged = shouldFlockingOccur?(accumulatedAbsoluteFlockingErrors)
      trainingSequence.nextEpoch
    end

    return mse, accumulatedAbsoluteFlockingErrors
  end

  def oneEpochOfAdaptingNetworkWeights(flockingHasConverged)
    accumulatedAbsoluteFlockingErrors = nil

    case flockingHasConverged
      when true
        accumulateOutputErrorDeltaWs
        adaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
        recenterEachNeuronsClusters(adaptingNeurons)    # necessary for later determining if clusters are still "tight-enough!"
        accumulatedAbsoluteFlockingErrors = []
      when false
        accumulatedAbsoluteFlockingErrors = accumulateFlockingErrorDeltaWs()
        adaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
    end

    mse = logNetworksResponses(adaptingNeurons)
    return mse, accumulatedAbsoluteFlockingErrors
  end

  def shouldFlockingOccur?(accumulatedAbsoluteFlockingErrors)
    displayFlockingErrorAndDeltaWInformation(accumulatedAbsoluteFlockingErrors)

    maxIterCount = maxFlockingIterationsCount
    maxIterCount = 0   if (tooEarlyToFlock?)

    needToMeasureOrReduceFlockingError = accumulatedAbsoluteFlockingErrors.empty? || ((accumulatedAbsoluteFlockingErrors.max / numberOfExamples) > args[:maxAbsFlockingErrorsPerExample])
    stillEnoughIterationsToFlock = flockingIterationsCount < maxIterCount
    if (needToMeasureOrReduceFlockingError && stillEnoughIterationsToFlock)
      self.flockingIterationsCount += 1
      return false
    else
      self.absFlockingErrorsOld = []
      self.flockingIterationsCount = 0
      return true
    end
  end

  def tooEarlyToFlock?
    return true if (trainingSequence.epochs < 200)
    return false
  end

  def useFuzzyClusters?
    exampleWeightings = adaptingNeurons.first.clusters[0].membershipWeightForEachExample
    criteria0 = 0.2
    criteria1 = 1.0 - criteria0
    count = 0
    exampleWeightings.each { |aWeight| count += 1 if (aWeight <= criteria0) }
    exampleWeightings.each { |aWeight| count += 1 if (aWeight > criteria1) }
    # puts "count=\t #{count}"
    return true if (count < numberOfExamples)
    return false
  end

  def displayFlockingErrorAndDeltaWInformation(accumulatedAbsoluteFlockingErrors)
    temp = absFlockingErrorsOld.deep_clone unless (absFlockingErrorsOld.nil?) # for display purposes only
    relativeChanges = deltaDispersions(accumulatedAbsoluteFlockingErrors)
    measuredAccumDeltaWs = (adaptingNeurons.collect { |aNeuron| aNeuron.inputLinks.collect { |aLink| aLink.deltaWAccumulated } })[0] # TODO this a quick and dirty measurement...
                                                                                                                                     # puts "absFlockingErrorsOld=\t#{temp}\tabsFlockingErrors=\t#{accumulatedAbsoluteFlockingErrors}\tfractionalChanges=\t#{relativeChanges}\tLinkO=\t#{measuredAccumDeltaWs[0]}\tLink1=\t#{measuredAccumDeltaWs[1]}\tLink2=\t#{measuredAccumDeltaWs[2]}\tPREVIOUS mse=\t#{network.calcNetworksMeanSquareError} "
  end

  def deltaDispersions(absFlockingErrors)
    unless (absFlockingErrors.nil? || absFlockingErrorsOld.nil? || absFlockingErrorsOld.empty?)
      index = -1
      fractionalChanges = absFlockingErrors.collect do |anAbsFlockingError|
        index += 1
        # relativeChange = (anAbsFlockingError - absFlockingErrorsOld[index]) / ((anAbsFlockingError + absFlockingErrorsOld[index])/2.0)
        fractionalChange = anAbsFlockingError / absFlockingErrorsOld[index]
      end
    else
      fractionalChanges = Array.new(adaptingNeurons.length) { nil }
    end
    self.absFlockingErrorsOld = absFlockingErrors
    return fractionalChanges
  end

  def accumulateOutputErrorDeltaWs
    adaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }
    acrossExamplesAccumulateDeltaWs { |aNeuron, dataRecord| aNeuron.calcAccumDeltaWsForOutputError }
  end

  def accumulateFlockingErrorDeltaWs
    adaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:flockingLearningRate] }
    adaptingNeurons.each { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError = 0.0 }
    acrossExamplesAccumulateDeltaWs do |aNeuron, dataRecord|
      dataRecord[:localFlockingError] = aNeuron.calcLocalFlockingError { aNeuron.weightedExamplesCenter } if (useFuzzyClusters?)
      dataRecord[:localFlockingError] = aNeuron.calcLocalFlockingError { aNeuron.centerOfDominantClusterForExample } unless (useFuzzyClusters?)
      aNeuron.calcAccumDeltaWsForLocalFlocking
    end
    adaptingNeurons.collect { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError }
  end

  def acrossExamplesAccumulateDeltaWs
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      adaptingNeurons.each do |aNeuron|
        dataRecord = aNeuron.recordResponsesForExample
        yield(aNeuron, dataRecord, exampleNumber)
      end
    end
  end

  def nameTrainingGroupsAndSetLearningRatesForStep1OfTraining
    self.layersWithInputLinks = [outputLayer]
    self.adaptingLayers = [outputLayer]
    self.layersWhoseClustersNeedToBeSeeded = [outputLayer]
    setUniversalNeuronGroupNames
  end

  def setUniversalNeuronGroupNames
    self.neuronsWithInputLinks = layersWithInputLinks.flatten
    self.neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
    self.allNeuronsInOneArray = inputLayer + neuronsWithInputLinks

    self.adaptingNeurons = adaptingLayers.flatten unless (adaptingLayers.nil?)
    self.neuronsWhoseClustersNeedToBeSeeded = layersWhoseClustersNeedToBeSeeded.flatten unless (layersWhoseClustersNeedToBeSeeded.nil?)
  end
end

class XORTrainer < SimpleAdjustableLearningRateTrainer

  def postInitialize
    super
    @hiddenLayer1 = network.hiddenLayer1
  end

  def step1NameTrainingGroupsAndLearningRates
    self.layersWithInputLinks = [hiddenLayer1 + outputLayer]
    self.adaptingLayers = layersWithInputLinks
    self.layersWhoseClustersNeedToBeSeeded = layersWithInputLinks
    setUniversalNeuronGroupNames
  end
end

###########  The following 3 Trainers use the old training versions...
class CircleTrainer < AbstractTrainer
  attr_accessor :learningRateNoFlockPhase1, :neuronsCreatingFlockingError, :neuronsAdaptingToLocalFlockingError,
                :learningRateLocalFlockPhase2, :neuronsAdaptingToBackPropedFlockingError, :learningRateForBackPropedFlockingErrorPhase2

  def nameTrainingGroupsAndLearningRates
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]
    self.neuronsCreatingFlockingError = outputLayer
    self.neuronsAdaptingToLocalFlockingError = outputLayer
    self.learningRateLocalFlockPhase2 = args[:learningRateLocalFlockPhase2]
    self.neuronsAdaptingToBackPropedFlockingError = hiddenLayer
    self.learningRateForBackPropedFlockingErrorPhase2 = args[:learningRateForBackPropedFlockingErrorPhase2]
  end

  def simpleLearningWithFlocking(examples, theNeuron = nil)
    distributeSetOfExamples(examples)
    mse = 99999.0
    dPrimes = nil
    @dPrimesOld = nil

    nameTrainingGroupsAndLearningRates()

    seedClustersInFlockingNeurons(neuronsCreatingFlockingError)
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      case trainingSequence.status
        when :inPhase1
          neuronsWithInputLinks.each { |neuron| neuron.learningRate = learningRateNoFlockPhase1 }
          mse = adaptNetworkNoFlocking
        when :inPhase2
          @dPrimesOld = nil if (trainingSequence.atStartOfPhase2)

          neuronsAdaptingToLocalFlockingError.each { |neuron| neuron.learningRate = learningRateLocalFlockPhase2 }
          neuronsAdaptingToBackPropedFlockingError.each { |neuron| neuron.learningRate = learningRateForBackPropedFlockingErrorPhase2 }

          mse, dPrimes = adaptNetworkWithFlockingOnly(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)

          #if (trainingSequence.epochs%400 == 0)
          #  testMSE = self.oneForwardPassEpoch(testingExamples)
          #  puts "epochs=\t#{trainingSequence.epochs}\tmse=\t#{mse}\ttestMSE=\t#{testMSE}"
          #end
          printIt = selectEpochsForPrinting(epochOffsets = [0, 1, 2, 5, 10], interval=100)

          deltaDPrimeRatios = deltaDPrimes(dPrimes)

        # puts "epochs=\t#{trainingSequence.epochs}\tdeltaDPrimeRatios.max=\t#{deltaDPrimeRatios.max}\tdeltaDPrimeRatios.min=\t#{deltaDPrimeRatios.min}\tdPrimes=\t#{dPrimes}" if (printIt)
        # trainingSequence.dPrimesAreLargeEnough if ((deltaDPrimeRatios.map{|e|e.abs}).max < 0.001)
        else
          STDERR.puts "status of training sequencer is not understood"
      end
      trainingSequence.nextEpoch
    end
    return trainingSequence.epochs, mse, dPrimes
  end

  def selectEpochsForPrinting(epochOffsets, interval = 40.0)
    # TODO    This is HIGHLY UGLY CODE
    epochs = trainingSequence.epochs
    quotient = (epochs / interval).to_f
    truncatedEpochs = quotient.floor * interval
    logicalValues = epochOffsets.map { |offset| ((epochs.to_f - (truncatedEpochs + offset)).abs < 0.00000001) }
    #puts "logicalValues=\t#{logicalValues}"
    printIt = false
    printIt = true if ((logicalValues.select { |e| e }).length > 0)
    #puts "printIt=\t#{printIt}"
    return printIt
  end

  protected

  ###------------   Adaption to Only Flocking Error; NOT!! Output Error  ------------------------

  def adaptNetworkWithFlockingOnly(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    measureAndStoreAllNeuralResponsesFlockingOnly(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
    dPrimes = recenterEachNeuronsClusters(neuronsCreatingFlockingError)
    mse = logNetworksResponses(neuronsCreatingFlockingError + neuronsAdaptingToLocalFlockingError)
    [mse, dPrimes]
  end

  def measureAndStoreAllNeuralResponsesFlockingOnly(neuronsCreatingFlockingError, neuronsAdaptingToLocalFlockingError, neuronsAdaptingToBackPropedFlockingError)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }

    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate } # only really need to do this for the lowest layer that will create flocking error
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }

      # measure flocking error created in those neurons, so designated!
      neuronsCreatingFlockingError.each do |aNeuron|
        dataRecorded = aNeuron.recordResponsesForExample
        localFlockingError = aNeuron.calcLocalFlockingError
        dataRecorded[:localFlockingError] = localFlockingError
        aNeuron.recordResponsesForExampleToDB(dataRecorded)
        aNeuron.backPropagate { |errorFromUpperLayers, localFlockError| localFlockError } # necessary for sending flocking error to hidden layer
      end

      # send flocking error to hidden layer
      neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.backPropagate }

      # accumulate weight changes in response to that flocking error so assigned/distributed
      neuronsAdaptingToBackPropedFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate }
      neuronsAdaptingToLocalFlockingError.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |errorFromUpperLayers, localFlockError| localFlockError } }
    end
  end

  ###------------ One Epoch Testing Section Section ------------------------------
  def measureNeuralResponsesForTesting
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      neuronsWithInputLinks.each { |aNeuron| aNeuron.recordResponsesForExample }
    end
    mse = network.calcNetworksMeanSquareError
  end
end

class Trainer4Class < AbstractTrainer
  attr_accessor :learningRateNoFlockPhase1, :neuronsCreatingLocalFlockingErrorAndAdaptingToSame,
                :learningRateLocalFlockPhase2, :neuronsAdaptingOnlyToBPOutputError, :learningRateBPOutputErrorPhase2

  def simpleLearningWithFlocking(examples, theNeuron = nil)
    distributeSetOfExamples(examples)
    mse = 99999.0
    dPrimes = nil
    @dPrimesOld = nil

    nameTrainingGroupsAndLearningRates()

    seedClustersInFlockingNeurons(neuronsCreatingLocalFlockingErrorAndAdaptingToSame)
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      case trainingSequence.status
        when :inPhase1
          neuronsWithInputLinks.each { |neuron| neuron.learningRate = learningRateNoFlockPhase1 }

          mse = adaptNetworkNoFlocking

        when :inPhase2
          @dPrimesOld = nil if (trainingSequence.atStartOfPhase2)

          neuronsCreatingLocalFlockingErrorAndAdaptingToSame.each { |neuron| neuron.learningRate = learningRateLocalFlockPhase2 }
          neuronsAdaptingOnlyToBPOutputError.each { |neuron| neuron.learningRate = learningRateBPOutputErrorPhase2 }

          mse, dPrimes = adaptNetworkWithLocalFlockingIncluded(neuronsCreatingLocalFlockingErrorAndAdaptingToSame, neuronsAdaptingOnlyToBPOutputError)

          deltaDPrimeRatios = deltaDPrimes(dPrimes)
          puts "epochs=\t#{trainingSequence.epochs}\tdeltaDPrimeRatios=\t#{deltaDPrimeRatios}" if (trainingSequence.epochs > 30)
        # trainingSequence.dPrimesHaveNotChangedMuch if (deltaDPrimeRatios.max < 0.01)
        else
          STDERR.puts "status of training sequencer is not understood"
      end
      trainingSequence.nextEpoch
    end
    return trainingSequence.epochs, mse, dPrimes
  end

  def nameTrainingGroupsAndLearningRates
    self.learningRateNoFlockPhase1 = args[:learningRateNoFlockPhase1]
    self.neuronsCreatingLocalFlockingErrorAndAdaptingToSame = hiddenLayer
    self.learningRateLocalFlockPhase2 = args[:learningRateLocalFlockPhase2]
    self.neuronsAdaptingOnlyToBPOutputError = neuronsWithInputLinks - neuronsCreatingLocalFlockingErrorAndAdaptingToSame
    self.learningRateBPOutputErrorPhase2 = args[:learningRateBPOutputErrorPhase2]
  end
end

class TunedTrainerAnalogy4ClassNoBPofFlockError < AbstractTrainer
  attr_accessor :layerTuners, :numberOfLayersWithInputLinks

  def initialize(trainingSequence, network, args)
    @trainingSequence = trainingSequence
    @network = network
    @args = args
    @dataStoreManager = SimulationDataStoreManager.instance
    @allNeuronLayers = network.createSimpleLearningANN
    @startTime = Time.now
    @elapsedTime = nil
    @minMSE = args[:minMSE]
    postInitialize
  end

  def postInitialize
    @inputLayer = network.inputLayer
    @hiddenLayer1 = network.hiddenLayer1
    @hiddenLayer2 = network.hiddenLayer2
    @hiddenLayer3 = network.hiddenLayer3
    @outputLayer = network.outputLayer
    @theBiasNeuron = network.theBiasNeuron
  end

  def simpleLearningWithFlocking(examples, arrayOfNeuronsToPlot)
    step1NameTrainingGroupsAndLearningRates
    mse, dPrimes = oneStepOfLearningAndDisplay(examples, arrayOfNeuronsToPlot)

    trainingSequence.nextStep
    step2NameTrainingGroupsAndLearningRates
    mse, dPrimes = oneStepOfLearningAndDisplay(examples, arrayOfNeuronsToPlot)

    40.times do |doubleStepNumber| # 800
      trainingSequence.nextStep
      step3NameTrainingGroupsAndLearningRates
      mse, dPrimes = oneStepOfLearningAndDisplay(examples, arrayOfNeuronsToPlot)
      # mse, dPrimes = stepLearning(examples)

      trainingSequence.nextStep
      step4NameTrainingGroupsAndLearningRates
      mse, dPrimes = oneStepOfLearningAndDisplay(examples, arrayOfNeuronsToPlot)
                                   # mse, dPrimes = stepLearning(examples)
    end

    return trainingSequence.epochs, mse, dPrimes
  end

  def step1NameTrainingGroupsAndLearningRates
    # -----   Adaption to OUTPUT Error  -------
    self.layersWithInputLinks = [hiddenLayer1, outputLayer]
    self.layersToAdaptToOutputError = [hiddenLayer1, outputLayer]

    # -----   Adaption to FLOCKING Error  -------
    self.layersCreatingFlockingError = [hiddenLayer1, outputLayer]
    self.layersAdaptingToLocalFlockingError = [hiddenLayer1, outputLayer]
    self.layersWhoseClustersNeedToBeSeeded = [hiddenLayer1, outputLayer]

    outputLayerTuner = LearningAndFlockingTuner.new(outputLayer, args, true, 0.08)
    hiddenLayer1Tuner = LearningAndFlockingTuner.new(hiddenLayer1, args, false, 0.08)

    self.layerTuners = [outputLayerTuner, hiddenLayer1Tuner]
    setUniversalNeuronGroupNames
  end

  def step2NameTrainingGroupsAndLearningRates
    # Expanding Network to 2 Hidden Layers
    disconnect_one_layer_from_another(hiddenLayer1, outputLayer)
    disconnect_one_layer_from_another([theBiasNeuron], outputLayer)
    connect_layer_to_another(hiddenLayer2, outputLayer, args)
    connect_layer_to_another([theBiasNeuron], outputLayer, args)

    # Because a layer has been added, succeeding layers weights should be reinitialized with random weights.
    outputLayer.each { |aNeuron| aNeuron.randomizeLinkWeights }

    # -----   Adaption to OUTPUT Error  ------
    self.layersWithInputLinks = [hiddenLayer1, hiddenLayer2, outputLayer]
    self.layersToAdaptToOutputError = [hiddenLayer2, outputLayer]

    # -----   Adaption to FLOCKING Error  -------
    self.layersCreatingFlockingError = [hiddenLayer2, outputLayer]
    self.layersAdaptingToLocalFlockingError = [hiddenLayer2, outputLayer]
    self.layersWhoseClustersNeedToBeSeeded = [hiddenLayer2, outputLayer]

    outputLayerTuner = LearningAndFlockingTuner.new(outputLayer, args, true, 0.08)
    hiddenLayer2Tuner = LearningAndFlockingTuner.new(hiddenLayer2, args, false, 0.08)

    self.layerTuners = [outputLayerTuner, hiddenLayer2Tuner]
    setUniversalNeuronGroupNames
  end

  def step3NameTrainingGroupsAndLearningRates
    # -----   Adaption to OUTPUT Error  ------
    self.layersWithInputLinks = [hiddenLayer1, hiddenLayer2, outputLayer]
    self.layersToAdaptToOutputError = [hiddenLayer1, outputLayer]

    # -----   Adaption to FLOCKING Error  -------
    self.layersCreatingFlockingError = [hiddenLayer1]
    self.layersAdaptingToLocalFlockingError = [hiddenLayer1]
    #self.layersWhoseClustersNeedToBeSeeded = [hiddenLayer1]

    outputLayerTuner = LearningAndFlockingTuner.new(outputLayer, args, noFlock = true, 0.08)
    hiddenLayer1Tuner = LearningAndFlockingTuner.new(hiddenLayer1, args, noFlock = false, 0.08)

    self.layerTuners = [outputLayerTuner, hiddenLayer1Tuner]
    setUniversalNeuronGroupNames
  end

  def step4NameTrainingGroupsAndLearningRates
    # -----   Adaption to OUTPUT Error  ------
    self.layersWithInputLinks = [hiddenLayer1, hiddenLayer2, outputLayer]
    self.layersToAdaptToOutputError = [hiddenLayer2, outputLayer]

    # -----   Adaption to FLOCKING Error  -------
    self.layersCreatingFlockingError = [hiddenLayer2]
    self.layersAdaptingToLocalFlockingError = [hiddenLayer2]
    #self.layersWhoseClustersNeedToBeSeeded = [hiddenLayer2]

    outputLayerTuner = LearningAndFlockingTuner.new(outputLayer, args, noFlock = true, 0.08)
    hiddenLayer2Tuner = LearningAndFlockingTuner.new(hiddenLayer2, args, noFlock = false, 0.08)

    self.layerTuners = [outputLayerTuner, hiddenLayer2Tuner]
    setUniversalNeuronGroupNames
  end
end

######################################################################

### Currently unused...
class WeightChangeNormalizer
  attr_accessor :layer, :weightChangeSetPoint

  def initialize(layer, args, weightChangeSetPoint = 0.08)
    @layer = layer
    @weightChangeSetPoint = weightChangeSetPoint
  end

  def normalizeWeightChanges
    layerGain = weightChangeSetPoint / maxWeightChange
    layer.each { |neuron| neuron.inputLinks.each { |aLink| aLink.deltaWAccumulated = layerGain * aLink.deltaWAccumulated } }
  end

  private

  def maxWeightChange
    acrossLayerMaxValues = layer.collect do |aNeuron|
      accumulatedDeltaWsAbs = aNeuron.inputLinks.collect { |aLink| aLink.deltaWAccumulated.abs }
      accumulatedDeltaWsAbs.max
    end
    return acrossLayerMaxValues.max
  end
end


### Currently very complex for its current use cases
class TrainingSequence
  attr_accessor :network, :args, :epochs, :epochsInPhase1, :epochsInPhase2, :numberOfEpochsInCycle,
                :maxNumberOfEpochs, :epochsSinceBeginningOfCycle, :status,
                :stillMoreEpochs, :lastEpoch, :inPhase2, :inPhase1,
                :atStartOfCycle, :atStartOfPhase1, :atStartOfPhase2,
                :atStartOfTraining, :afterFirstEpoch, :dataStoreManager,
                :tagDataSet, :epochDataSet, :numberOfEpochsBetweenStoringDBRecords

  private_class_method(:new)

  @@trainingSequence = nil

  def TrainingSequence.create(network, args)
    @@trainingSequence = new(network, args) unless @@trainingSequence
    @@trainingSequence
  end

  def TrainingSequence.instance
    return @@trainingSequence
  end

  def initialize(network, args)
    @network = network
    @args = args
    @dataStoreManager = SimulationDataStoreManager.instance
    @tagDataSet = dataStoreManager.tagDataSet
    @epochDataSet = dataStoreManager.epochDataSet

    @epochsInPhase1 = args[:phase1Epochs]
    @epochsInPhase2 = args[:phase2Epochs]
    @numberOfEpochsInCycle = epochsInPhase1 + epochsInPhase2
    @maxNumberOfEpochs = args[:maxNumEpochs]
    @numberOfEpochsBetweenStoringDBRecords = @args[:numberOfEpochsBetweenStoringDBRecords]

    @epochs = -1
    @epochsSinceBeginningOfCycle = -1
    @atStartOfTraining = true
    @afterFirstEpoch = false
    @stillMoreEpochs = true
    @lastEpoch = false
    @status = :inPhase1
    nextEpoch
  end

  def startAtBeginningOfCycle
    @epochsSinceBeginningOfCycle = -1
    @status = :inPhase1
  end

  def dPrimesAreLargeEnough
    startAtBeginningOfCycle
  end

  def nextEpoch
    storeSequenceTagData

    self.epochs += 1
    self.atStartOfTraining = false if (epochs > 0)
    self.afterFirstEpoch = true unless (atStartOfTraining)

    self.epochsSinceBeginningOfCycle += 1
    self.epochsSinceBeginningOfCycle = 0 if (epochsSinceBeginningOfCycle == numberOfEpochsInCycle)

    self.atStartOfCycle = false
    self.atStartOfPhase1 = false
    if (epochsSinceBeginningOfCycle == 0)
      self.atStartOfCycle = true
      self.atStartOfPhase1 = true if (epochsInPhase1 > 0)
    end

    self.inPhase1 = false
    self.inPhase1 = true if (epochsSinceBeginningOfCycle < epochsInPhase1)

    self.atStartOfPhase2 = false
    self.atStartOfPhase2 = true if ((epochsSinceBeginningOfCycle == epochsInPhase1) && (epochsInPhase2 > 0))

    self.inPhase2 = false
    self.inPhase2 = true if (epochsSinceBeginningOfCycle >= epochsInPhase1)

    self.lastEpoch = false
    self.lastEpoch = true if (epochs == (maxNumberOfEpochs - 1))

    self.stillMoreEpochs = true
    self.stillMoreEpochs = false if (epochs >= maxNumberOfEpochs)

    self.status = :inPhase1 if (inPhase1 == true)
    self.status = :inPhase2 if (inPhase2 == true)

    dataStoreManager.epochNumber = epochs
  end

  def timeToRecordData
    record = false
    return if (epochs < 0)
    record = true if ((epochs % numberOfEpochsBetweenStoringDBRecords) == 0)
    record = true if lastEpoch
    return record
  end

  def nextStep
    self.maxNumberOfEpochs = epochs + args[:maxNumEpochs]
    self.epochsSinceBeginningOfCycle = -1
    self.atStartOfTraining = true
    self.afterFirstEpoch = false
    self.stillMoreEpochs = true
    self.lastEpoch = false
    self.status = :inPhase1
    nextEpoch
    self.atStartOfTraining = true # TODO this is a fudge...
    self.afterFirstEpoch = true unless (atStartOfTraining)
  end

  private

  def storeSequenceTagData
    if (timeToRecordData)
      epochsSinceBeginningOfPhase = nil
      if (inPhase1)
        learningPhase = 'phase1'
        epochsSinceBeginningOfPhase = epochsSinceBeginningOfCycle
      else
        learningPhase = 'phase2'
        epochsSinceBeginningOfPhase = epochsSinceBeginningOfCycle - epochsInPhase1
      end
      dataToStore = {:epochNumber => epochs, :experimentNumber => dataStoreManager.theNumberOfTheCurrentExperiment,
                     :mse => network.mse, :learningPhase => learningPhase,
                     :epochsSinceBeginningOfPhase => epochsSinceBeginningOfPhase,
                     :epochsSinceBeginningOfCycle => epochsSinceBeginningOfCycle}
      tagDataSet.insert(dataToStore)
    end
  end
end

