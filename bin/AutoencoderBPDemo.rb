### VERSION "nCore"
## ../nCore/bin/AutocoderBPDemo.rb
# Purpose:  Simple backprop autocoder implementation. -- to build-up to and compare with flocking version.

require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'

srand(0)

numberOfExamples = 16

def createTrainingSet(numberOfExamples)
  include ExampleDistribution


  xStart = [-1.0, 1.0, -1.0, 1.0]
  yStart = [1.0, 1.0, -1.0, -1.0]
  xInc = [0.0, 0.0, 0.0, 0.0]
  yInc = [1.0, 1.0, -1.0, -1.0]


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


####################################################################
args = {:learningRate => 0.003, #1.0,
        :weightRange => 1.0,
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 2,
        :numberOfOutputNeurons => 2,
        :numberOfExamples => numberOfExamples,
}

aLearningNetwork = AutocoderNetwork.new(dataStorageManager=nil, args)
allNeuronLayers = aLearningNetwork.createSimpleLearningANN

allNeuronsInOneArray = allNeuronLayers.flatten
inputLayer = allNeuronLayers[0]
hiddenLayer = allNeuronLayers[1]
outputLayer = allNeuronLayers[2]
neuronsWithInputLinks = hiddenLayer + outputLayer
neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse

# create the training examples...
examples = createTrainingSet(numberOfExamples)
distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
# puts examples

mse = 99999.0
epochNumber = 0
while  (mse > 0.01 && epochNumber < 10**5)
  neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  numberOfExamples.times do |exampleNumber|
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.recordResponsesForExample }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate }
  end
  neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }

  mse = aLearningNetwork.calcNetworksMeanSquareError
  if (epochNumber.modulo(100) == 0)
    puts "At Epoch # #{epochNumber} Network's MSE=\t#{aLearningNetwork.calcNetworksMeanSquareError}\n\n" if (epochNumber.modulo(100) == 0)
    aLearningNetwork.recordResponse(epochNumber)
  end

  epochNumber += 1
end

puts "At Epoch # #{epochNumber} Network's MSE=\t#{mse}\n\n"

puts aLearningNetwork

mseVsEpochMeasurements = aLearningNetwork.measures
x = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:epochNumber] }
y = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:networkMSE] }
aPlotter = Plotter.new(title="2 Hidden Neuron BP Autocoder", "Number of Epochs", "Error on Training Set Error", aFilename = nil)
aPlotter.plot(x, y)