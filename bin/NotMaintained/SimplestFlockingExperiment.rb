### VERSION "nCore"
## ../nCore/bin/SimplestFlockingExperiment.rb
# Purpose:  To explore methods for improving the accuracy and speed of flocking, -- and its
# "interleaving with" sop backpropagation of output errors

require 'C:/Code/DevRuby/NN2012/nCore/lib/core/DataSet'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NeuralParts'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NeuralPartsExtended'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NetworkFactories'
require 'C:/Code/DevRuby/NN2012/nCore/lib/plot/CorePlottingCode'

def embedExampleDataInto(inputLayer, outputLayer)
  include ExampleDistribution

  examples = []
  examples << {:inputs => [1.0, 1.0], :targets => [1.0], :exampleNumber => 0}
  examples << {:inputs => [1.0, 2.0], :targets => [1.0], :exampleNumber => 1}
  examples << {:inputs => [1.0, 3.0], :targets => [1.0], :exampleNumber => 2}
  examples << {:inputs => [1.0, 4.0], :targets => [1.0], :exampleNumber => 3}
  examples << {:inputs => [-1.0, -1.0], :targets => [0.0], :exampleNumber => 4}
  examples << {:inputs => [-1.0, -2.0], :targets => [0.0], :exampleNumber => 5}
  examples << {:inputs => [-1.0, -3.0], :targets => [0.0], :exampleNumber => 6}
  examples << {:inputs => [-1.0, -4.0], :targets => [0.0], :exampleNumber => 7}
  distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
end

def reportMetrics(outputNeuron, epochNumber, aLearningNetwork, dataArray, args)
  oneEpochsMeasures = outputNeuron.metricRecorder.withinEpochMeasures
  dataArray << (oneEpochsMeasures).dup if ((epochNumber+1).modulo(1) == 0)
  mse = aLearningNetwork.calcNetworksMeanSquareError
  aLearningNetwork.recordResponse(mse, epochNumber)
  if (epochNumber.modulo(250) == 0)
    puts "At Epoch # #{epochNumber} Network's MSE=\t#{aLearningNetwork.calcNetworksMeanSquareError}\n"
    theFlockLearningRate = args[:flockLearningRate]
    oneEpochsMeasures.each_with_index do |measuresForAnExample, exampleNumber|
      # std("measuresForAnExample",measuresForAnExample)
      puts "ex #{exampleNumber}\tBP Error=\t#{measuresForAnExample[:error]}\tFlocking Error=\t#{theFlockLearningRate * measuresForAnExample[:localFlockingError]}"
    end
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


####################################################################
srand(0)
numberOfExamples = 8

# You can greatly speed things up (but with a penalty of poorer flocking) by increasing
# bpLearningRate and flockLearningRate by 100.  Then 'correcting' flockLearningRate by
# reducing it by 5-10
args = {:learningRate => 1.0, #1.0,
        :bpLearningRate => 0.3, # 0.003, #0.3,
        :weightRange => 1.0,
        :numberOfInputNeurons => 2,
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
        :flockLearningRate => -0.01 # -0.0009  #-0.01  # use '0.0' instead to illustrate sop backprop without flocking!
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

# include the training examples...
embedExampleDataInto(inputLayer, outputLayer)

dataArray = []
mse = 99999.0
epochNumber = 0

epochsOfFlockingOnly = 5

def learningWithMomentumAndFocusedFlocking(aLearningNetwork, allNeuronsInOneArray, args, dataArray, epochNumber, epochsOfFlockingOnly, flockingNeurons, mse, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer)
  interval = epochsOfFlockingOnly + 1
  while (mse > 0.001)
    neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
    numberOfExamples.times do |exampleNumber|
      allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
      neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate { |bpError, flockError| bpError } }
      outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
      neuronsWithInputLinks.each { |aNeuron| aNeuron.recordExampleMeasurements }
      flockingNeurons.each { |aNeuron| aNeuron.calcLocalFlockingError } if (epochNumber > 0)
      if (epochNumber.modulo(interval) == 0)

        neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |bpError, flockError|
          bpError + aNeuron.augmentFlockingErrorUsingMomentum(flockError, alpha = 0.8, exampleNumber) } }
      else
        neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |bpError, flockError|
          aNeuron.augmentFlockingErrorUsingMomentum(flockError, alpha = 0.8, exampleNumber) } }
      end
    end
    neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
    flockingNeurons.each { |aNeuron| aNeuron.clusterAllResponses }

    mse = reportMetrics(outputLayer[0], epochNumber, aLearningNetwork, dataArray, args)
    epochNumber += 1
  end
  return epochNumber, mse
end

epochNumber, mse = learningWithMomentumAndFocusedFlocking(aLearningNetwork, allNeuronsInOneArray, args, dataArray, epochNumber, epochsOfFlockingOnly, flockingNeurons, mse, neuronsWithInputLinks, neuronsWithInputLinksInReverseOrder, numberOfExamples, outputLayer)

plotMSEvsEpochNumber(aLearningNetwork)

plotFlockingErrorVsEpochNumber(dataArray)

puts "At Epoch # #{epochNumber} Network's MSE=\t#{mse}\n\n"

puts aLearningNetwork # display neural network's final state -- after training is complete.



