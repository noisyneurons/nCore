### VERSION "nCore"
## ../nCore/bin/BasicBPDemo.rb
## Simple backprop demo. For XOR, and given parameters, requires 2080 epochs to converge.
##                       For OR, and given parameters, requires 166 epochs to converge.

require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/plot/CorePlottingCode'

include ExampleDistribution

srand(0) # For a '0' argument normally takes 6053 epochs to reach error criteria of 0.01

# add the training examples...
examples = []
examples << {:inputs => [0.0, 0.0], :targets => [0.1], :exampleNumber => 0}
examples << {:inputs => [0.0, 1.0], :targets => [0.9], :exampleNumber => 1}
examples << {:inputs => [1.0, 0.0], :targets => [0.9], :exampleNumber => 2}
examples << {:inputs => [1.0, 1.0], :targets => [0.1], :exampleNumber => 3}

numberOfExamples = examples.length

####################################################################
args = {:learningRate => 1.0,
        :weightRange => 1.0,
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 3,
        :numberOfOutputNeurons => 1,
        :numberOfExamples => numberOfExamples
}

aLearningNetwork = LearningNetwork.new(dataStorageManager=nil, args)
allNeuronLayers = aLearningNetwork.createSimpleLearningANN

puts "Bias Neuron output= #{aLearningNetwork.theBiasNeuron.output}"

allNeuronsInOneArray = allNeuronLayers.flatten
inputLayer = allNeuronLayers[0]
hiddenLayer = allNeuronLayers[1]
outputLayer = allNeuronLayers[2]

neuronsWithInputLinks = hiddenLayer + outputLayer
neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse

distributeDataToInputAndOutputNeurons(examples, [allNeuronLayers.first, allNeuronLayers.last])

mse = 1.0
epochNumber = 0
while (mse > 0.01)
  neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  numberOfExamples.times do |exampleNumber|
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate }
    outputLayer.each { |aNeuron| aNeuron.recordResponsesForExample }
  end
  neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
  mse = aLearningNetwork.calcNetworksMeanSquareError

  if (epochNumber.modulo(100) == 0)
    puts "At Epoch # #{epochNumber} Network's MSE=\t#{aLearningNetwork.calcNetworksMeanSquareError}\n\n" if (epochNumber.modulo(100) == 0)
    aLearningNetwork.recordResponses
  end

  epochNumber += 1
end
puts "At Epoch # #{epochNumber} Network's MSE=\t#{mse}\n\n"

#puts aLearningNetwork

mseVsEpochMeasurements = aLearningNetwork.measures
x = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:epochNumber] }
y = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:networkMSE] }
aPlotter = Plotter.new
aPlotter.plot(x, y)