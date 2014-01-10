### VERSION "nCore"
## ../nCore/bin/OneInOneOutBP.rb
# Purpose:  2-Class classifier, 2 gaussian clusters with NonMon IO function for simple 1 in 1 out network

require_relative 'BaseLearningExperiment'

#class Neuron
#  # include NonMonotonicIOFunctionUnShifted
#  include NonMonotonicIODerivative
#end

class OutputNeuron
  include NonMonotonicIODerivative
end


class Experiment
  include NeuronToNeuronConnection

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.1,
        :minMSE => 0.00001, # 0.001,
        :maxNumEpochs => 2e3, # 6e3,

        # Network Architecture
        :numberOfInputNeurons => 1,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => Link,

        # Training Set parameters
        :numberOfExamples => (self.numberOfExamples = 16),
        :numberOfTestingExamples => (numberOfExamples * 1000),

        # Recording and database parameters
        :neuronToDisplay => 2,
        :intervalForSavingNeuronData => 100,
        :intervalForSavingDetailedNeuronData => 1000,
        :intervalForSavingTrainingData => 100
    }
  end

  def createTrainingSet
    createSimplest2GaussianClustersForTrainAndTest(numberOfExamples)
  end

  def createTestingSet
    createSimplest2GaussianClustersForTrainAndTest(args[:numberOfTestingExamples])
  end

  def createNetworkAndTrainer
    network = Simplest1LayerNet.new(args)
    theTrainer = BPTrainingSupervisorFor1LayerNet.new(examples, network, args)
    return network, theTrainer
  end

  private

  def createSimplest2GaussianClustersForTrainAndTest(numberOfExamples)
    examples = []
    numberOfClasses = 2
    numberOfExamplesInEachClass = numberOfExamples / numberOfClasses

    negativeNormalCluster = NormalDistribution.new(-1.0, 0.3)
    positiveNormalCluster = NormalDistribution.new(1.0, 0.3)

    arrayOfClassIndexAndClassRNGs = []

    classIndexAndClassRNG = [(classIndex = 0), (classRNG = negativeNormalCluster)]
    arrayOfClassIndexAndClassRNGs << classIndexAndClassRNG
    classIndexAndClassRNG = [(classIndex = 1), (classRNG = positiveNormalCluster)]
    arrayOfClassIndexAndClassRNGs << classIndexAndClassRNG

    exampleNumber = 0
    arrayOfClassIndexAndClassRNGs.each do |classIndexAndClassRNG|
      classIndex = classIndexAndClassRNG[0]
      desiredOutputs = [classIndex.to_f]
      numberOfExamplesInEachClass.times do |classExNumb|

        randomNumberGenerator = classIndexAndClassRNG[1]
        x = randomNumberGenerator.rng
        aPoint = [x]

        examples << {:inputs => aPoint, :targets => desiredOutputs, :exampleNumber => exampleNumber, :class => classIndex}
        exampleNumber += 1
      end
    end
    STDERR.puts "cross-check failed on: 'number of examples'" if (examples.length != (numberOfExamplesInEachClass * numberOfClasses))
    examples
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0

experiment = Experiment.new("NonMon1in1out   2-Class classifier, 2 gaussian clusters with NonMon IO function for simple 1 in 1 out network", baseRandomNumberSeed)

experiment.performSimulation()