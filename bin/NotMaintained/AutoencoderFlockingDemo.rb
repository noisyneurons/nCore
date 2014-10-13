### VERSION "nCore"
## ../nCore/bin/AutoencoderFlockingDemo.rb
# Purpose:  Simple backprop autocoder implementation. -- to build-up to and compare with flocking version.

require 'C:/Code/DevRuby/NN2012/nCore/lib/core/DataSet'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NeuralParts'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NeuralPartsExtended'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NetworkFactories'
require 'C:/Code/DevRuby/NN2012/nCore/lib/plot/CorePlottingCode'

srand(0)

numberOfExamples = 16

def createTrainingSet(numberOfExamples)
  include ExampleDistribution


  xStart = [-1.0, 1.0, -1.0, 1.0]
  yStart = [1.0, 1.0, -1.0, -1.0]
  xInc = [0.0, 0.0, 0.0, 0.0]
  yInc = [1.0, 1.0, -1.0, -1.0]

  # target = [0.9, 0.9, 0.9, 0.9] # potentially will use this example set for supervised learning.

  numberOfClasses = xStart.length
  numberOfExamplesInEachClass = numberOfExamples / numberOfClasses
  exampleNumber = 0
  xIncrement = 0.0
  yIncrement = 0.0
  examples = []
  numberOfClasses.times do |indexToClass|
    xS = xStart[indexToClass]
    xI = xInc[indexToClass]
    yS = yStart[indexToClass]
    yI = yInc[indexToClass]
    numberOfExamplesInEachClass.times do |classExNumb|
      x = xS + (xI * classExNumb)
      y = yS + (yI * classExNumb)
      aPoint = [x, y]
      examples << {:inputs => aPoint, :targets => aPoint, :exampleNumber => exampleNumber}
      exampleNumber += 1
    end
  end
  STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  examples
end

#def reportMetrics(neuronOfInterest, epochNumber, aLearningNetwork, dataArray, args)
#  dataArray << (neuronOfInterest.metricRecorder.withinEpochMeasures).dup if ((epochNumber+1).modulo(1) == 0)
#  mse = aLearningNetwork.calcNetworksMeanSquareError
#  aLearningNetwork.recordResponse(mse, epochNumber)
#  oneEpochsMeasures = neuronOfInterest.metricRecorder.withinEpochMeasures
#  if (epochNumber.modulo(25000) == 0)
#    puts "At Epoch # #{epochNumber} Network's MSE=\t#{aLearningNetwork.calcNetworksMeanSquareError}\n"
#    theFlockLearningRate = args[:flockLearningRate]
#    oneEpochsMeasures.each_with_index do |measuresForAnExample, exampleNumber|
#      puts "ex #{exampleNumber}\tBP Error=\t#{measuresForAnExample[:error]}\tFlocking Error=\t#{theFlockLearningRate * measuresForAnExample[:localFlockingError]}"
#    end
#  end
#  mse
#end
#
#def plotMSEvsEpochNumber(aLearningNetwork)
#  mseVsEpochMeasurements = aLearningNetwork.measures
#  x = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:epochNumber] }
#  y = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:networkMSE] }
#  aPlotter = Plotter.new(title="Simplest Flocking Training Error", "Number of Epochs", "Error on Training Set", aFilename = nil)
#  aPlotter.plot(x, y)
#end
#
#def plotFlockingErrorVsEpochNumber(dataArray)
#  arrayOfFlockingErrors = []
#  arrayOfEpochNumbers = []
#  dataArray.each do |oneEpochsMeasures|
#    oneEpochsMeasures.each do |measuresForAnExample|
#      arrayOfEpochNumbers << (measuresForAnExample[:epochNumber]).to_f
#      arrayOfFlockingErrors << measuresForAnExample[:localFlockingError]
#    end
#  end
#
#  aPlotter2 = Plotter.new(title="Simplest Flocking FlkError", "Number of Epochs", "Flocking Error", aFilename = nil)
#  aPlotter2.plot(arrayOfEpochNumbers, arrayOfFlockingErrors)
#end

####################################################################
args = {:learningRate => 0.003, #1.0,
        :bpLearningRate => 1.0,
        :weightRange => 1.0,
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 2,
        :numberOfOutputNeurons => 2,
        :numberOfExamples => numberOfExamples,
        # Parameters associated with flocking...
        :typeOfClusterer => DynamicClusterer,
        :numberOfClusters => 2,
        :m => 2.0,
        :numExamples => numberOfExamples,
        :exampleVectorLength => 2,
        :delta => 0.001,
        :maxNumberOfClusteringIterations => 100,
        :flockLearningRate => -0.1 # 0.0#
}

aLearningNetwork = AutocoderFlockingNetwork.new(args)
allNeuronLayers = aLearningNetwork.createSimpleLearningANN

allNeuronsInOneArray = allNeuronLayers.flatten
inputLayer = allNeuronLayers[0]
hiddenLayer = allNeuronLayers[1]
outputLayer = allNeuronLayers[2]
neuronsWithInputLinks = hiddenLayer + outputLayer
neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
flockingNeurons = hiddenLayer

# create the training examples...
examples = createTrainingSet(numberOfExamples)
distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
# puts examples

dataArray = []

mse = 99999.0
epochNumber = 0
epochSwitch = -1

while (mse > 0.01 && epochNumber < 10**5)

  #flockingNeurons.each { |aNeuron| aNeuron.flockLearningRate = 0.0 } if (epochNumber < epochSwitch)
  #flockingNeurons.each { |aNeuron| aNeuron.flockLearningRate = args[:flockLearningRate] } if (epochNumber >= epochSwitch)

  #flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 } if (epochNumber >= epochSwitch)

  neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  numberOfExamples.times do |exampleNumber|
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate { |bpError, flockError| bpError } }
    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.recordExampleMeasurements }
    flockingNeurons.each { |aNeuron| aNeuron.calcLocalFlockingError } if (epochNumber > 0)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |bpError, flockError| bpError + flockError } }
  end
  neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
  flockingNeurons.each { |aNeuron| aNeuron.clusterAllResponses } ## Required for calculation of flocking error on subsequent epoch
  # std("epoch number ", epochNumber)
  mse = aLearningNetwork.calcNetworksMeanSquareError
  if (epochNumber.modulo(100) == 0)
    puts "At Epoch # #{epochNumber} Network's MSE=\t#{aLearningNetwork.calcNetworksMeanSquareError}\n\n" # if (epochNumber.modulo(100) == 0)
    aLearningNetwork.recordResponse(mse, epochNumber)
  end

  #mse = reportMetrics(hiddenLayer[0], epochNumber, aLearningNetwork, dataArray, args)
  #puts "At Epoch # #{epochNumber} Network's MSE=\t#{mse}\n\n"

  epochNumber += 1
end

puts "At Epoch # #{epochNumber} Network's MSE=\t#{mse}\n\n"

puts aLearningNetwork # display neural network's final state -- after training is complete.


#plotMSEvsEpochNumber(aLearningNetwork)
#
#plotFlockingErrorVsEpochNumber(dataArray)

