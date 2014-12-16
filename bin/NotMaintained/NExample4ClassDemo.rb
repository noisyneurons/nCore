### VERSION "nCore"
## ../nCore/bin/NExample4ClassDemo.rb
# Purpose:  Simple backprop autocoder implementation. -- to build-up to and compare with flocking version.
# Primary experimental parameters: flockLearningRate (0.0 vs. -0.03), epochSwitch (-1 vs. 1000)
# Secondary experimental parameters:  bpLearningRate, learningRate

require 'C:/Code/DevRuby/NN2012/nCore/lib/core/DataSet'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NeuralParts'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NeuralPartsExtended'
require 'C:/Code/DevRuby/NN2012/nCore/lib/core/NetworkFactories'
require 'C:/Code/DevRuby/NN2012/nCore/lib/plot/CorePlottingCode'

srand(0)

numberOfExamples = 16

def create4ClassTrainingSet(numberOfExamples, rightShiftUpper2Classes = 0.0)
  include ExampleDistribution

  xStart = [-1.0, 1.0, -1.0, 1.0]
  yStart = [1.0, 1.0, -1.0, -1.0]


  xInc = [0.0+rightShiftUpper2Classes, 0.0+rightShiftUpper2Classes, 0.0, 0.0]
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
      targets = [0.0, 0.0, 0.0, 0.0]
      targets[indexToClass] = 1.0
      examples << {:inputs => aPoint, :targets => targets, :exampleNumber => exampleNumber}
      exampleNumber += 1
    end
  end
  logger.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
  examples
end


####################################################################
args = {:learningRate => 0.03, #1.0,
        :bpLearningRate => 1.0,
        :weightRange => 1.0,
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 2,
        :numberOfOutputNeurons => 4,
        :numberOfExamples => numberOfExamples,
        # Parameters associated with flocking...
        :typeOfClusterer => DynamicClusterer,
        :numberOfClusters => 2,
        :m => 2.0,
        :numExamples => numberOfExamples,
        :exampleVectorLength => 2,
        :delta => 0.001,
        :maxNumberOfClusteringIterations => 100,
        :flockLearningRate => -0.03 # -0.3 # -0.1
}

# aLearningNetwork = NExample4ClassFeedbackNetwork.new(args)
aLearningNetwork = NExample4ClassNetwork.new(args)
allNeuronLayers = aLearningNetwork.createSimpleLearningANN

allNeuronsInOneArray = allNeuronLayers.flatten
inputLayer = allNeuronLayers[0]
hiddenLayer = allNeuronLayers[1]
outputLayer = allNeuronLayers[2]
neuronsWithInputLinks = hiddenLayer + outputLayer
neuronsWithInputLinksInReverseOrder = neuronsWithInputLinks.reverse
flockingNeurons = hiddenLayer

# create the training examples...
examples = create4ClassTrainingSet(numberOfExamples, rightShiftUpper2Classes = 0.0)
distributeDataToInputAndOutputNeurons(examples, [inputLayer, outputLayer])
# logger.puts examples

dataArray = []

mse = 99999.0
epochNumber = 0
epochSwitch = -1

while (mse > 0.01 && epochNumber < 10**4)

  if (epochSwitch > 0)
    #flockingNeurons.each { |aNeuron| aNeuron.flockLearningRate = 0.0 } if (epochNumber < epochSwitch)
    #flockingNeurons.each { |aNeuron| aNeuron.flockLearningRate = args[:flockLearningRate] } if (epochNumber >= epochSwitch)
    flockingNeurons.each { |aNeuron| aNeuron.bpLearningRate = 0.0 } if (epochNumber >= epochSwitch)
  end

  neuronsWithInputLinks.each { |aNeuron| aNeuron.zeroDeltaWAccumulated }
  neuronsWithInputLinks.each { |aNeuron| aNeuron.clearWithinEpochMeasures }
  numberOfExamples.times do |exampleNumber|
    allNeuronsInOneArray.each { |aNeuron| aNeuron.propagate(exampleNumber) }
    neuronsWithInputLinksInReverseOrder.each { |aNeuron| aNeuron.backPropagate { |bpError, flockError| bpError } }
    outputLayer.each { |aNeuron| aNeuron.calcWeightedErrorMetricForExample }
    neuronsWithInputLinks.each { |aNeuron| aNeuron.recordExampleMeasurements }

    if (epochNumber > 0)
      flockingNeurons.each do |aNeuron|
        aNeuron.calcLocalFlockingError do |localFE, arg2, arg3|
          logger.puts "localFE\t#{localFE}\targ2\t#{arg2}\targ3\t#{arg3}" if (epochNumber.modulo(500) == 0)
        end
      end
    end

#    neuronsWithInputLinks.each { |aNeuron| aNeuron.calcDeltaWsAndAccumulate { |bpError, flockError| bpError + flockError } }
    neuronsWithInputLinks.each do |aNeuron|
      aNeuron.calcDeltaWsAndAccumulate do |bpError, flockError, unAdjFlockingError, flockLearningRate|
        logger.puts "epochNumber\t#{epochNumber}\texampleNumber\t#{exampleNumber}\tNeurons id\t#{aNeuron.id}\tbpError\t#{bpError}\tflockError\t#{flockError}\tunAdjFlockingError\t#{unAdjFlockingError}\tflockLearningRate\t#{flockLearningRate}" if (epochNumber.modulo(500) == 0)
        bpError + flockError
      end
    end
  end
  neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
  flockingNeurons.each { |aNeuron| aNeuron.clusterAllResponses } ## Required for calculation of flocking error on subsequent epoch
  # std("epoch number ", epochNumber)
  mse = aLearningNetwork.calcNetworksMeanSquareError
  if (epochNumber.modulo(100) == 0)
    logger.puts "At Epoch # #{epochNumber} Network's MSE=\t#{aLearningNetwork.calcNetworksMeanSquareError}\n\n" # if (epochNumber.modulo(100) == 0)
    aLearningNetwork.recordResponse(mse, epochNumber)
  end

#mse = reportMetrics(hiddenLayer[0], epochNumber, aLearningNetwork, dataArray, args)
#logger.puts "At Epoch # #{epochNumber} Network's MSE=\t#{mse}\n\n"

  epochNumber += 1
end

logger.puts "At Epoch # #{epochNumber} Network's MSE=\t#{mse}\n\n"

logger.puts aLearningNetwork # display neural network's final state -- after training is complete.


#plotMSEvsEpochNumber(aLearningNetwork)
#
#plotFlockingErrorVsEpochNumber(dataArray)

