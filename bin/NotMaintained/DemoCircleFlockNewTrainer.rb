# DemoCircleFlockNewTrainer.rb

require './lib/WeightedFlocking'

flockFlag = true

class Trainer
  def train # two phases of training: backprop and then flocking -- on an epoch by epoch basis
    srand @randomSeed
    @epochNumber = 0

    # setupDifferentFlockLearningRatesAcrossFlockingLayer    # TODO this is cool and should be documented

    beginningOfTrainingMeasures
    @maxNumberOfEpochs.times do |epochIndex|
      @epochNumber = epochIndex
      beginningOfEpochProcessingMeasures
      backProp()
      break if @trainingMSE < @stoppingErrorCriteria ##?? This may not be the best location for stopping training
      collectDataForInitialClustering()
      @network.useClusteringToDetermineFlocks(randomizeFlockCenters = true) # Here we determine and assign clusters to each
      if (@numberOfFlockingIterations > 0)
        iterativeFlocking()
      end
      collectDataForInitialClustering()
      @network.useClusteringToDetermineFlocks(randomizeFlockCenters = true) # Here we determine and assign clusters to each
      endOfFlockingMeasures
      endOfEpochProcessingMeasures
    end
    endOfTrainingMeasures
    return @epochNumber
  end

  def iterativeFlocking
    @flockIterationNumber = nil
    beginningOfFlockingMeasures
    (@numberOfFlockingIterations).times do |flockIterationNumber|
      @flockIterationNumber = flockIterationNumber
      @network.setFlockDeltaWAccumulated(0.0)
      @network.initWithinEpochMetrics
      @exampleList.each_with_index do |example, exampleNumber|
        @network.driveNetworkWithExample(example)
        @network.propagate
        @network.backPropagate
        @network.zeroExamplesFE
        @network.calculateExamplesLocalFE(exampleNumber) # this occurs within the neuron and NOT the links.
        @network.backPropagateFE # This BP of FE only occurs for the hidden neuron, because this function is disabled in the output neuron.
        @network.calcLinksFDeltaWAndAccumulate
        endOfExampleProcessingMeasures
      end
      @network.addFlockingAccumulationToWeight
    end
  end

  def collectDataForInitialClustering
    @network.initWithinEpochMetrics
    @exampleList.each_with_index do |example, exampleNumber|
      @network.driveNetworkWithExample(example)
      @network.propagate
      @network.backPropagate
      @network.storeNeuronsInputAndBPError
    end
  end

  def backProp
    totalSquaredError = 0.0
    #@network.initWithinEpochMetrics # TODO "beginningOfEpochMethod" poorly named.
    @exampleList.each do |example|
      @network.driveNetworkWithExample(example)
      @network.propagate
      errorSquaredForExample = @evaluationLayerForMetrics.networksErrorOnExample
      # std("example Error= ",errorSquaredForExample)
      totalSquaredError += errorSquaredForExample

      #squaredError += @evaluationLayer.networksErrorOnExample
      @network.backPropagate
      #@network.storeNeuronsInputAndBPError
      @network.calcDeltaWAndAccumulate
    end
    @network.addAccumulationToWeight
    @trainingMSE = totalSquaredError / (@numExamples * @numOutputs)
    std("trainingMSE= ", @trainingMSE) if (@epochNumber.modulo(@specification.numberOfEpochsBetweenMeasures) == 0)
    @network.setDeltaWAccumulated(0.0)
  end

  def setupDifferentFlockLearningRatesAcrossFlockingLayer
    @network.layersForLocalFlocking.each do |aFlockingLayer|
      numberOfNeurons = aFlockingLayer.nnComponents.length
      numberOfNeuronsWithFlocking = numberOfNeurons / 2
      aFlockingLayer.nnComponents.each_with_index do |aNeuron, neuronsIndex|
        if (neuronsIndex < numberOfNeuronsWithFlocking)
          aNeuron.setFlockLearningRate(@specification.flockLearningRate)
        else
          aNeuron.setFlockLearningRate(0.0)
        end
      end
    end
  end
end

################## Generation of Examples on 2 Concentric Circles ##########################

def generateExamplesOnConcentricCircles(firstExamplesAngleToXAxis, args)
  numberOfExamples = args[:numberOfExamples]

  numberOfClasses = 2
  numExamplesPerClass = numberOfExamples / numberOfClasses

  angleBetweenExamplesInDegrees = 360.0 / numExamplesPerClass
  radiusArray = [1.0, 1.3]
  desiredOutput = [0.0, 1.0]

  examples = []
  numExamplesPerClass.times do |exampleNumberWithinClass|
    angle = (angleBetweenExamplesInDegrees * exampleNumberWithinClass) + firstExamplesAngleToXAxis
    angleInRadians = angle * (360.0/(2.0 * Math::PI))
    numberOfClasses.times do |indexToClass|
      radius = radiusArray[indexToClass]
      x = radius * Math.cos(angleInRadians)
      y = radius * Math.sin(angleInRadians)
      aPoint = [x, y]
      targets = [desiredOutput[indexToClass]]
      exampleNumber = if (indexToClass == 1)
                        exampleNumberWithinClass + numExamplesPerClass
                      else
                        exampleNumberWithinClass
                      end
      anExample = {:inputs => aPoint, :targets => targets, :exampleNumber => exampleNumber, :class => indexToClass}
      examples << anExample
    end
  end
  return examples
end

exampleList = generateExamplesOnConcentricCircles(firstExamplesAngleToXAxis = 0.0)
testingExampleList = generateExamplesOnConcentricCircles(firstExamplesAngleToXAxis = 15.0)

############################################
# Specify parameters for training
# generic Training Specifications:
specification = NNSpecification.new
specification.plotSubdirectory = "DemoCircleFlock"
specification.exampleList = exampleList
specification.weightRange = 1.0 #
specification.stoppingErrorCriteria = 0.01 #
specification.maxNumberOfEpochs = 50000 #
specification.randomSeed = 1 #
specification.numberOfClusters = 2 #
specification.numberOfEpochsBetweenMeasures = 200 #
specification.m = 2.0 #
specification.numExamples = 24 #
specification.exampleVectorLength = 2 #


if (flockFlag == true)
  specification.experimentName = "DemoCircleWithSoftFlocking"
  specification.learningRate = 0.03 #
  specification.flockLearningRate = -0.001 # -0.01     #
  specification.numberOfFlockingIterations = 10 # 10   #
else
  specification.experimentName = "DemoCircleNOSoftFlocking"
  specification.learningRate = 0.1
  specification.flockLearningRate = 0.0
  specification.numberOfFlockingIterations = 1
end

Neuron.zeroID
Layer.zeroID

srand(specification.randomSeed)

inputLayer = LayerOfInputNeurons.new(2, specification)
hiddenLayer1 = Layer.new(inputLayer, NeuronThatBackPropsFEAndChangesWeightsButDoesntCalcLocalFE, 10, specification)
outputLayer = Layer.new(hiddenLayer1, NeuronCalcsLocalFEbutNoWeightChanges, 1, specification)
evaluationLayer = EvaluationLayer.new(outputLayer, specification)
allLayersExceptEvaluationLayer = [inputLayer, hiddenLayer1, outputLayer]

# Training configuration.
networkToTest = Network.new(allLayersExceptEvaluationLayer, evaluationLayer, specification)
trainerOfNetwork = Trainer.new(specification.exampleList, networkToTest, specification)

numberOfEpochsBetweenMeasures = specification.numberOfEpochsBetweenMeasures
indexToOutputLayer=2
indexToNeuronInOutputLayer=0
networksIOContourPlot = ContourPlotOfIOFunction.new(trainerOfNetwork, numberOfEpochsBetweenMeasures, indexToOutputLayer, indexToNeuronInOutputLayer, specification)
outputNeuronsBPandFlockError = BothErrorsVsNetInputAndFlockingViewClusterPlots.new(trainerOfNetwork, numberOfEpochsBetweenMeasures, indexToOutputLayer, indexToNeuronInOutputLayer, specification)
indexToHiddenLayer=1
indexToNeuronInHiddenLayer=0
bpAndFlockErrorVsNetInput = BothErrorsVsNetInputAndFlockingViewClusterPlots.new(trainerOfNetwork, numberOfEpochsBetweenMeasures, indexToHiddenLayer, indexToNeuronInHiddenLayer, specification)
tester = Tester.new(testingExampleList, trainerOfNetwork, numberOfEpochsBetweenMeasures, specification)
trainerOfNetwork.trainingAndMeasurementTriggers << outputNeuronsBPandFlockError << networksIOContourPlot << bpAndFlockErrorVsNetInput << tester

trainerOfNetwork.train

puts networkToTest
puts "Error Criteria Reached at Epoch: #{trainerOfNetwork.epochNumber}\t with a Training MSE = #{trainerOfNetwork.trainingMSE}"