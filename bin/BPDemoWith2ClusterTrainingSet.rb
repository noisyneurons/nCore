### VERSION "nCore"
## ../nCore/bin/BPDemoWith2ClusterTrainingSet.rb

require_relative  '../lib/core/DataSet'
require_relative  '../lib/core/NeuralParts'
require_relative  '../lib/core/NetworkFactories'
require_relative  '../lib/plot/CorePlottingCode'

include ExampleDistribution

srand(0)

# add the training examples...
examples = []
examples << {:inputs => [1.0, 1.0], :targets => [1.0], :exampleNumber => 0}
examples << {:inputs => [1.0, 2.0], :targets => [1.0], :exampleNumber => 1}
examples << {:inputs => [1.0, 3.0], :targets => [1.0], :exampleNumber => 2}
examples << {:inputs => [1.0, 4.0], :targets => [1.0], :exampleNumber => 3}
examples << {:inputs => [-1.0, -1.0], :targets => [0.0], :exampleNumber => 4}
examples << {:inputs => [-1.0, -2.0], :targets => [0.0], :exampleNumber => 5}
examples << {:inputs => [-1.0, -3.0], :targets => [0.0], :exampleNumber => 6}
examples << {:inputs => [-1.0, -4.0], :targets => [0.0], :exampleNumber => 7}

numberOfExamples = examples.length

####################################################################
args = {:learningRate => 0.3, #1.0,
        :weightRange => 1.1, # 1.0,
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 0,
        :numberOfOutputNeurons => 1,
        :numberOfExamples => numberOfExamples
}

aLearningNetwork = LearningNetwork.new(args)
allNeuronLayers = aLearningNetwork.createSimpleLearningANN

puts "Bias Neuron output= #{aLearningNetwork.theBiasNeuron.output}"

allNeuronsInOneArray = allNeuronLayers.flatten
inputLayer = allNeuronLayers[0]
outputLayer = allNeuronLayers[1]
neuronsWithInputLinks = outputLayer
neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse

distributeDataToInputAndOutputNeurons(examples, [allNeuronLayers.first, allNeuronLayers.last])

mse = 1.0
epochNumber = 0
while (mse > 0.001)
  neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  numberOfExamples.times do |exampleNumber|
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate }
    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate }
    outputLayer.each { |aNeuron| aNeuron.recordExampleMeasurements }
  end
  neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
  mse = aLearningNetwork.calcNetworksMeanSquareError

  if (epochNumber.modulo(1) == 0)
    puts "At Epoch # #{epochNumber} Network's MSE=\t#{aLearningNetwork.calcNetworksMeanSquareError}\n\n" if (epochNumber.modulo(100) == 0)
    aLearningNetwork.recordResponse(epochNumber)
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

