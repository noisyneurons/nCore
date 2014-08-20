### VERSION "nCore"
## ../nCore/lib/core/Trainers.rb


#####

class TrainingSequence
  attr_accessor :args, :epochs, :maxNumberOfEpochs,
                :stillMoreEpochs, :lastEpoch,
                :atStartOfTraining, :afterFirstEpoch

  def initialize(args)
    @args = args
    @maxNumberOfEpochs = args[:maxNumEpochs]
    @epochs = -1
    @atStartOfTraining = true
    @afterFirstEpoch = false
    @stillMoreEpochs = true
    @lastEpoch = false
    nextEpoch
  end

  def epochs=(value)
    @epochs = value
    args[:epochs] = value
  end

  def nextEpoch
    self.epochs += 1
    self.atStartOfTraining = false if (epochs > 0)
    self.afterFirstEpoch = true unless (atStartOfTraining)

    self.lastEpoch = false
    self.lastEpoch = true if (epochs == (maxNumberOfEpochs - 1))

    self.stillMoreEpochs = true
    self.stillMoreEpochs = false if (epochs >= maxNumberOfEpochs)
  end
end



###################################################################
###################################################################

class TrainerBase
  attr_accessor :examples, :network, :numberOfOutputNeurons, :allNeuronLayers, :allNeuronsInOneArray, :inputLayer, :outputLayer,
                :neuronsWithInputLinks, :neuronsWithInputLinksInReverseOrder, :args, :trainingSequence,
                :numberOfExamples, :startTime, :elapsedTime, :minMSE
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
    @neuronsWithInputLinks = (@allNeuronsInOneArray - @inputLayer).flatten
    @neuronsWithInputLinksInReverseOrder =  @neuronsWithInputLinks.reverse
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
    mse = 1e100
    while ((mse >= minMSE) && trainingSequence.stillMoreEpochs)
      performStandardBackPropTraining()
      neuronsWithInputLinks.each { |aNeuron| aNeuron.dbStoreNeuronData }
      dbStoreTrainingData()
      trainingSequence.nextEpoch
      mse = calcMSE
    end
    testMSE = calcTestingMeanSquaredErrors
    return trainingSequence.epochs, calcMSE, testMSE
  end

  def performStandardBackPropTraining
    #puts  neuronsWithInputLinks
    neuronsWithInputLinks.each { |aNeuron| aNeuron.learningRate = args[:outputErrorLearningRate] }
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
    allNeuronsInOneArray.flatten.each { |aNeuron| aNeuron.propagate(exampleNumber) }
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

