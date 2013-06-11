### VERSION "nCore"
## ../nCore/bin/ProbabilityOf1ClusterVsTheOther.rb
# Purpose:  To quantitively explore the simplest clustering w/o supervision.

demoToPerform = "flockingWithNOSupervision"
# Choices are:  flockingWithNOSupervision, flockingWithMomentum, withMomentumAndFocusedFlocking

require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NeuralPartsExtended'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'

#  SIMPLEST DATASET --> One input dimension training set with 2 unit separation between nearest exemplars of the 2 classes
def createTrainingSet(numberOfExamples, separationBetweenDataPoints)
  include ExampleDistribution

  xStart = [-1.0, 1.0]
  xInc = [(-1.0*separationBetweenDataPoints), separationBetweenDataPoints]

  # target = [0.9, 0.9, 0.9, 0.9] # potentially will use this example set for supervised learning.

  numberOfClasses = xStart.length
  numberOfExamplesInEachClass = numberOfExamples / numberOfClasses
  exampleNumber = 0
  xIncrement = 0.0
  examples = []
  numberOfClasses.times do |indexToClass|
    xS = xStart[indexToClass]
    xI = xInc[indexToClass]
    numberOfExamplesInEachClass.times do |classExNumb|
      x = xS + (xI * classExNumb)
      aPoint = [x]
      examples << {:inputs => aPoint, :targets => [indexToClass.to_f], :exampleNumber => exampleNumber}
      exampleNumber += 1
    end
  end
  STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  examples
end

def reportMetrics(outputNeuron, epochNumber, aLearningNetwork, dataArray, args)
  oneEpochsMeasures = outputNeuron.metricRecorder.withinEpochMeasures
  dataArray << (oneEpochsMeasures).dup if ((epochNumber+1).modulo(1) == 0)
  mse = aLearningNetwork.calcNetworksMeanSquareError
  aLearningNetwork.recordResponse(mse, epochNumber)
  if (epochNumber.modulo(5) == 0)
    puts "At Epoch # #{epochNumber} Network's MSE=\t#{aLearningNetwork.calcNetworksMeanSquareError}\n"
    theFlockLearningRate = args[:flockLearningRate]
    #oneEpochsMeasures.each_with_index do |measuresForAnExample, exampleNumber|
    #  # std("measuresForAnExample",measuresForAnExample)
    #  puts "ex #{exampleNumber}\tBP Error=\t#{measuresForAnExample[:error]}\tFlocking Error=\t#{theFlockLearningRate * measuresForAnExample[:localFlockingError]}"
    #end
  end
  mse
end

def plotMSEvsEpochNumber(aLearningNetwork)
  mseVsEpochMeasurements = aLearningNetwork.measures
  x = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:epochNumber] }
  y = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:networkMSE] }
  aPlotter = Plotter.new(title="Simplest Flocking Training Error", "Number of Epochs", "Error on Training Set", aFilename = nil)
  aPlotter.plot(x, y)
end

def plotFlockingErrorVsEpochNumber(dataArray)
  arrayOfFlockingErrors = []
  arrayOfEpochNumbers = []
  dataArray.each_with_index do |oneEpochsMeasures, epochNumber|
    oneEpochsMeasures.each do |measuresForAnExample|
      arrayOfEpochNumbers << epochNumber.to_f
      arrayOfFlockingErrors << measuresForAnExample[:localFlockingError]
    end
  end

  aPlotter2 = Plotter.new(title="Simplest Flocking FlkError", "Number of Epochs", "Flocking Error", aFilename = nil)
  aPlotter2.plot(arrayOfEpochNumbers, arrayOfFlockingErrors)
end

def simpleLearningWithFlocking(epochSwitch, aLearningNetwork, allNeuronsInOneArray, args, dataArray, epochNumber, flockingNeurons, mse, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer)

  while (mse > 0.001 && epochNumber<3.0*10**3)

    if (epochSwitch > 0)
      flockingNeurons.each { |aNeuron| aNeuron.flockLearningRate = 0.0 } if (epochNumber < epochSwitch)
      flockingNeurons.each { |aNeuron| aNeuron.flockLearningRate = args[:flockLearningRate] } if (epochNumber >= epochSwitch)
      flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 } if (epochNumber >= epochSwitch)
      puts aLearningNetwork if (epochNumber == epochSwitch)
    end

    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate { |bpError, flockError| bpError } }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      neuronsWithInputLinks.each { |aNeuron| aNeuron.recordExampleMeasurements }
      flockingNeurons.each { |aNeuron| aNeuron.calcLocalFlockingError { |dataForReport| periodicallyDisplayContentsOfHash(dataForReport, epochNumber, interval=epochSwitch) } } if (epochNumber > 0)
      neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |bpError, flockError| bpError + flockError } }
    end
    neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
    flockingNeurons.each { |aNeuron| aNeuron.initializeClusterCenters } if (epochNumber == 0)
    flockingNeurons.each { |aNeuron| aNeuron.clusterAllResponses }

    mse = reportMetrics(outputLayer[0], epochNumber, aLearningNetwork, dataArray, args)
    epochNumber += 1
  end
  return epochNumber, mse
end

####################################################################
srand(0)
numberOfExamples = 8

args = {:learningRate => 1.0,
        :bpLearningRate => 0.1,
        :weightRange => 0.01,
        :numberOfInputNeurons => 1,
        :numberOfHiddenNeurons => 0,
        :numberOfOutputNeurons => 1,
        :numberOfExamples => numberOfExamples,

        # Parameters associated with flocking...
        :typeOfClusterer => DynamicClusterer,
        :numberOfClusters => 2,
        :m => 2.0,
        :numExamples => numberOfExamples,
        :exampleVectorLength => 2,
        :delta => 0.001,
        :maxNumberOfClusteringIterations => 100,
        :flockLearningRate => -0.01
}

# Specify network
aLearningNetwork = AllFlockingNeuronNetwork.new(args)
allNeuronLayers = aLearningNetwork.createSimpleLearningANN

allNeuronsInOneArray = allNeuronLayers.flatten
inputLayer = allNeuronLayers[0]
outputLayer = allNeuronLayers[1]
neuronsWithInputLinks = outputLayer
neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
flockingNeurons = outputLayer

############################### include these 1-D training examples...
separationBetweenDataPoints = 0.01
examples = createTrainingSet(numberOfExamples, separationBetweenDataPoints)
distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
################################

dataArray = []
mse = 99999.0
epochNumber = 0
epochSwitch = 500

case demoToPerform
  when "flockingWithNOSupervision"
    epochNumber, mse = simpleLearningWithFlocking(epochSwitch, aLearningNetwork, allNeuronsInOneArray, args, dataArray, epochNumber, flockingNeurons, mse, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer)
  else
    STDERR.puts "did not understand CHOICE!"
end

plotMSEvsEpochNumber(aLearningNetwork)
plotFlockingErrorVsEpochNumber(dataArray)
puts "At Epoch # #{epochNumber} Network's MSE=\t#{mse}\n\n"
puts aLearningNetwork # display neural network's final state -- after training is complete.
