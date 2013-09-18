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
    trainingSequence.epochs < 200
  end
end

###########...

class AbstractStepTrainer
  attr_accessor :examples, :numberOfExamples, :neuronGroups, :trainingSequence,
                :args, :numberOfOutputNeurons,

                :allNeuronLayers, :inputLayer, :outputLayer,
                :layersWithInputLinks, :adaptingLayers,
                :layersWhoseClustersNeedToBeSeeded,

                :allNeuronsInOneArray, :neuronsWithInputLinks,
                :neuronsWithInputLinksInReverseOrder, :adaptingNeurons,
                :neuronsWhoseClustersNeedToBeSeeded,

                :maxFlockingIterationsCount, :flockingIterationsCount, :accumulatedAbsoluteFlockingErrors,
                :accumulatedExampleImportanceFactors, :absFlockingErrorsOld


  include ExampleDistribution
  include FlockingDecisionRoutines

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
    @adaptingLayers = neuronGroups.adaptingLayers
    @adaptingNeurons = neuronGroups.adaptingNeurons
    @layersWhoseClustersNeedToBeSeeded = neuronGroups.layersWhoseClustersNeedToBeSeeded
    @neuronsWhoseClustersNeedToBeSeeded = neuronGroups.neuronsWhoseClustersNeedToBeSeeded
  end

  def train
    STDERR.puts "Error:  called abstract method"
  end

  def prepareForRepeatedFlockingIterations
    recenterEachNeuronsClusters(adaptingNeurons) # necessary for later determining if clusters are still "tight-enough!"
    adaptingNeurons.each { |aNeuron| aNeuron.inputLinks.each { |aLink| aLink.store = 0.0 } }
    self.accumulatedAbsoluteFlockingErrors = []
  end

  #### Standard Backprop of Output Errors

  def performStandardBackPropTraining
    adaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }
    acrossExamplesAccumulateDeltaWs { |aNeuron, dataRecord| aNeuron.calcAccumDeltaWsForOutputError }
    adaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
  end

#### Local Flocking

  def adaptToLocalFlockError
    adaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:flockingLearningRate] }
    adaptingNeurons.each { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError = 0.0 }
    acrossExamplesAccumulateFlockingErrorDeltaWs()
    adaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
    self.accumulatedAbsoluteFlockingErrors = adaptingNeurons.collect { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError }
  end

  def acrossExamplesAccumulateFlockingErrorDeltaWs
    acrossExamplesAccumulateDeltaWs do |aNeuron, dataRecord|
      dataRecord[:localFlockingError] = aNeuron.calcLocalFlockingError { aNeuron.weightedExamplesCenter } if (useFuzzyClusters?)
      dataRecord[:localFlockingError] = aNeuron.calcLocalFlockingError { aNeuron.centerOfDominantClusterForExample } unless (useFuzzyClusters?)
      aNeuron.calcAccumDeltaWsForLocalFlocking
    end
  end

#### Used for BOTH Standard Backprop and Local Flocking

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
        aNeuron.dbStoreDetailedData
      end
    end
  end

  #### Refinements and specialized control functions:

  def useFuzzyClusters? # TODO Would this be better put 'into each neuron'
    exampleWeightings = adaptingNeurons.first.clusters[0].membershipWeightForEachExample # TODO Why is only the first neuron used for this determination?
    criteria0 = 0.2
    criteria1 = 1.0 - criteria0
    count = 0
    exampleWeightings.each { |aWeight| count += 1 if (aWeight <= criteria0) }
    exampleWeightings.each { |aWeight| count += 1 if (aWeight > criteria1) }
                                                                                         # puts "count=\t #{count}"
    return true if (count < numberOfExamples)
    return false
  end

  ###------------  Core Support Section ------------------------------

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
end

class FlockStepTrainer < AbstractStepTrainer

  def train(trials)
    distributeSetOfExamples(examples)
    seedClustersInFlockingNeurons(neuronsWhoseClustersNeedToBeSeeded)
    self.accumulatedAbsoluteFlockingErrors = []
    trials.times do
      innerTrainingLoop()
      dbStoreTrainingData
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
    adaptingNeurons.each { |aNeuron| aNeuron.dbStoreNeuronData }
  end
end


class TrainingSupervisor
  attr_accessor :stepTrainer, :trainingSequence, :network, :args, :startTime, :elapsedTime, :minMSE
  include RecordingAndPlottingRoutines

  def initialize(examples, network, args)
    @trainingSequence = args[:trainingSequence]
    @network = network
    @args = args
    @startTime = Time.now
    @elapsedTime = nil
    @minMSE = args[:minMSE]
    neuronGroups = NeuronGroupsSimplest.new(network)
    @stepTrainer = FlockStepTrainer.new(examples, neuronGroups, trainingSequence, args)
  end

  def train
    mse = 1e20
    numTrials = 1000
    while ((mse > minMSE) && trainingSequence.stillMoreEpochs)
      mse, accumulatedAbsoluteFlockingErrors = stepTrainer.train(numTrials)
    end
    arrayOfNeuronsToPlot = [network.outputLayer[0]]
    displayTrainingResults(arrayOfNeuronsToPlot)
    return trainingSequence.epochs, mse, accumulatedAbsoluteFlockingErrors
  end

  def displayTrainingResults(arrayOfNeuronsToPlot)
    puts network
    generatePlotForEachNeuron(arrayOfNeuronsToPlot) if arrayOfNeuronsToPlot.present?
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

