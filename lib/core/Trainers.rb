### VERSION "nCore"
## ../nCore/lib/core/Trainers.rb


module FlockingDecisionRoutines # TODO If one neuron does NOT meet criteria, all flocking neurons to continue to flock. Why not just those neurons that have not met the criteria.

  def flockingShouldOccur?(accumulatedAbsoluteFlockingErrors)
    return doNotFlock() if (tooEarlyToFlock?)
    # displayFlockingErrorAndDeltaWInformation(accumulatedAbsoluteFlockingErrors)
    if (doWeNeedToFlockNOW?(accumulatedAbsoluteFlockingErrors))
      return yesDoFlock()
    else
      return doNotFlock()
    end
  end

  def doWeNeedToFlockNOW?(accumulatedAbsoluteFlockingErrors)
    if (accumulatedAbsoluteFlockingErrors.empty?)
      true
    else
      accumulatedAbsoluteFlockingErrors = accumulatedAbsoluteFlockingErrors.delete_if{ |number| number.nan?}
      largestAbsoluteFlockingErrorPerExample = accumulatedAbsoluteFlockingErrors.max / numberOfExamples
      needToReduceFlockingError = largestAbsoluteFlockingErrorPerExample > args[:maxAbsFlockingErrorsPerExample]
      stillEnoughIterationsToFlock = flockingIterationsCount < maxFlockingIterationsCount
      (needToReduceFlockingError && stillEnoughIterationsToFlock)
    end
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
    args[:epochs] < 200
  end
end

###########...

class AbstractStepTrainer
  attr_accessor :examples, :numberOfExamples, :neuronGroups, :trainingSequence,
                :args, :numberOfOutputNeurons,

                :allNeuronLayers, :inputLayer, :outputLayer,
                :layersWithInputLinks, :layersWhoseClustersNeedToBeSeeded,

                :allNeuronsInOneArray, :neuronsWithInputLinks,
                :neuronsWithInputLinksInReverseOrder, :neuronsWhoseClustersNeedToBeSeeded,

                :outputErrorAdaptingLayers, :flockErrorGeneratingLayers, :flockErrorAdaptingLayers,
                :outputErrorAdaptingNeurons, :flockErrorGeneratingNeurons, :flockErrorAdaptingNeurons,

                :maxFlockingIterationsCount, :flockingIterationsCount, :accumulatedAbsoluteFlockingErrors,
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
    @numberOfExamples = examples.length

    @flockingIterationsCount = 0
    @maxFlockingIterationsCount = args[:maxFlockingIterationsCount]
    @accumulatedExampleImportanceFactors = nil
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
    STDERR.puts "Error:  called abstract method"
  end

  def performStandardBackPropTraining
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }
    acrossExamplesAccumulateDeltaWs(outputErrorAdaptingNeurons) { |aNeuron, dataRecord| aNeuron.calcAccumDeltaWsForHigherLayerError }
    outputErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
  end

  def prepareForRepeatedFlockingIterations
    recenterEachNeuronsClusters(flockErrorGeneratingNeurons)
    zeroOutFlockingLinksMomentumMemoryStore
    self.accumulatedAbsoluteFlockingErrors = []
  end

  def adaptToLocalFlockError
    STDERR.puts "Generating neurons and adapting neurons are not one in the same.  This is NOT local flocking!!" if (flockErrorGeneratingNeurons != flockErrorAdaptingNeurons)
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:flockingLearningRate] }
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError = 0.0 } # accumulatedAbsoluteFlockingError is a metric used for global control and monitoring
    acrossExamplesAccumulateFlockingErrorDeltaWs
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
    self.accumulatedAbsoluteFlockingErrors = flockErrorGeneratingNeurons.collect { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError }
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

  ###------------  Core Support Section ------------------------------
  #### Also, some refinements and specialized control functions:

  def calcMSE
    sse = outputLayer.inject(0.0) { |sum, anOutputNeuron| sum + anOutputNeuron.calcSumOfSquaredErrors }
    return (sse / (numberOfOutputNeurons * numberOfExamples))
  end

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

  def zeroOutFlockingLinksMomentumMemoryStore
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.inputLinks.each { |aLink| aLink.store = 0.0 } }
  end

  def calcNeuronsLocalFlockingError(aNeuron)
    localFlockingError = if (useFuzzyClusters?)
                           aNeuron.calcLocalFlockingError { aNeuron.weightedExamplesCenter }
                         else
                           aNeuron.calcLocalFlockingError { aNeuron.centerOfDominantClusterForExample }
                         end
  end

  def useFuzzyClusters? # TODO Would this be better put 'into each neuron'
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
    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
  end
end


class StepTrainerForLocalFlocking < AbstractStepTrainer
  def train(trials)
    distributeSetOfExamples(examples)
    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded)
    self.accumulatedAbsoluteFlockingErrors = []
    trials.times do
      innerTrainingLoop()
      dbStoreTrainingData()
      trainingSequence.nextEpoch
    end
    return calcMSE, accumulatedAbsoluteFlockingErrors
  end

  def innerTrainingLoop
    unless (flockingShouldOccur?(accumulatedAbsoluteFlockingErrors))
      performStandardBackPropTraining()
      prepareForRepeatedFlockingIterations()
    else
      adaptToLocalFlockError()
    end
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.dbStoreNeuronData }
  end
end


class BPofFlockingStepTrainer < StepTrainerForLocalFlocking
  def innerTrainingLoop
    unless (flockingShouldOccur?(accumulatedAbsoluteFlockingErrors))
      performStandardBackPropTraining()
      prepareForRepeatedFlockingIterations()
    else
      adaptToBPofFlockError
      std("h", [args[:epochs], accumulatedAbsoluteFlockingErrors])
    end
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.dbStoreNeuronData }
  end

  def adaptToBPofFlockError
    STDERR.puts "Generating neurons and adapting neurons are the same. Wrong Routine Called!!" if (flockErrorGeneratingNeurons == flockErrorAdaptingNeurons)
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:flockingLearningRate] }
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError = 0.0 }
    acrossExamplesAccumulateNonLocalFlockingErrorDeltaWs
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
    self.accumulatedAbsoluteFlockingErrors = flockErrorGeneratingNeurons.collect { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError }
  end

  def acrossExamplesAccumulateNonLocalFlockingErrorDeltaWs
    clearEpochAccumulationsInAllNeurons()
    numberOfExamples.times do |exampleNumber|
      propagateAcrossEntireNetwork(exampleNumber)
      backpropagateAcrossEntireNetwork()
      calcWeightedErrorMetricForExample()

      flockErrorGeneratingNeurons.each do |aNeuron|
        dataRecord = aNeuron.recordResponsesForExample
        dataRecord[:localFlockingError] = calcNeuronsLocalFlockingError(aNeuron)
        aNeuron.backPropagate { |higherLayerError, localFlockingError| localFlockingError } # TODO some unnecessary calculations.  Should change!!
        aNeuron.dbStoreDetailedData
      end

      flockErrorAdaptingNeurons.each do |aNeuron|
        aNeuron.backPropagate
        aNeuron.calcAccumDeltaWsForHigherLayerError
      end
    end
  end

  #def useFuzzyClusters? # TODO revert back to super...
  #  return true
  #end
end


class TrainingSupervisorBase
  attr_accessor :examples, :network, :args, :stepTrainer, :trainingSequence, :startTime, :elapsedTime, :minMSE
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
    numTrials = 1000
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

class TrainingSupervisorOutputNeuronLocalFlocking < TrainingSupervisorBase

  def postInitialize
    neuronGroups = NeuronGroupsTrivial.new(network)
    @stepTrainer = StepTrainerForLocalFlocking.new(examples, neuronGroups, trainingSequence, args)
  end
end

class TrainingSupervisorHiddenNeuronLocalFlocking   < TrainingSupervisorBase
  attr_accessor :stepTrainer, :trainingSequence, :neuronGroups, :network, :args, :startTime, :elapsedTime, :minMSE
  include RecordingAndPlottingRoutines

  def postInitialize
    @neuronGroups = NeuronGroupsHiddenNeuronLocalFlockingError.new(network)
    @stepTrainer = StepTrainerForLocalFlocking.new(examples, neuronGroups, trainingSequence, args)
  end
end

class TrainingSupervisorAllLocalFlockingLayers   < TrainingSupervisorBase
  attr_accessor :stepTrainer, :trainingSequence, :neuronGroups, :network, :args, :startTime, :elapsedTime, :minMSE
  include RecordingAndPlottingRoutines

  def postInitialize
    @neuronGroups = NeuronGroupsAllLocalFlockingLayers.new(network)
    @stepTrainer = StepTrainerForLocalFlocking.new(examples, neuronGroups, trainingSequence, args)
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

