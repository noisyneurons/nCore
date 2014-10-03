### VERSION "nCore"
## ../nCore/bin/OneInOneOutBP.rb
# Purpose:  2-Class classifier, 2 gaussian clusters with NonMon IO function for simple 1 in 1 out network

require_relative 'BaseLearningExperiment'


class OutputNeuron
  #include NonMonotonicIOFunction
  #include PiecewiseLinNonMonIOFunction

  def calcWeightedErrorMetricForExample
    oe = 1.0
    oe = 0.0 if (outputError.abs < 0.5)
    self.weightedErrorMetric = oe * oe # Assumes squared error criterion
  end
end


class Experiment
  include NeuronToNeuronConnection

  def setParameters
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => descriptionOfExperiment,
        :randomNumberSeed => randomNumberSeed,

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.001,
        :minMSE => 0.0, # 0.001,
        :maxNumEpochs => 6e3, # 6e3,

        # Network Architecture
        :numberOfInputNeurons => 1,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => Link,

        # Training Set parameters
        :dataSTD => 1.0,
        :numberOfExamples => (self.numberOfExamples = 16),
        :numberOfTestingExamples => (numberOfExamples * 1000),

        # Recording and database parameters
        :neuronToDisplay => 1,
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
    self.network = Simplest1LayerNet.new(args)
    self.theTrainer = BPTrainingSupervisorFor1LayerNet.new(examples, network, args)
    return network, theTrainer
  end

  protected

  def displayLastSnapShotRecords
    #keysToRecords = SnapShotData.lookup { |q| q[:experimentNumber].gte(0).order(:desc).limit(2) }
    #unless (keysToRecords.empty?)
    #  puts
    #  puts "Number\tLastEpoch\t\tTrainMSE\t\t\tTestMSE\t\t\tAccumulatedAbsoluteFlockingErrors\t\t\t\tTime\t\tTaskID\t\t\t\t\tDescription"
    #  keysToRecords.each do |keyToOneRecord|
    #    begin
    #      recordHash = SnapShotData.values(keyToOneRecord)
    #      puts "#{recordHash[:experimentNumber]}\t\t#{recordHash[:epochs]}\t\t#{recordHash[:trainMSE]}\t\t#{recordHash[:testMSE]}\t\t\t#{recordHash[:accumulatedAbsoluteFlockingErrors]}\t\t\t#{recordHash[:time]}\t\t\t#{recordHash[:gridTaskID]}\t\t\t\t#{recordHash[:descriptionOfExperiment]}"
    #    rescue
    #      puts "problem in yaml conversion"
    #    end
    #  end
    #  # recordHash = SnapShotData.values(keysToRecords.last)
    #end
  end

  def createSimplest2GaussianClustersForTrainAndTest(numberOfExamples)
    theCreatedExamples = []
    numberOfClasses = 2
    numberOfExamplesInEachClass = numberOfExamples / numberOfClasses

    negativeNormalCluster = NormalDistribution.new(-1.0, args[:dataSTD])
    positiveNormalCluster = NormalDistribution.new(1.0, args[:dataSTD])

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

        theCreatedExamples << {:inputs => aPoint, :targets => desiredOutputs, :exampleNumber => exampleNumber, :class => classIndex}
        exampleNumber += 1
      end
    end
    STDERR.puts "cross-check failed on: 'number of theCreatedExamples'" if (theCreatedExamples.length != (numberOfExamplesInEachClass * numberOfClasses))
    theCreatedExamples
  end
end


###################################### START of Main Learning  ##########################################

baseRandomNumberSeed = 0
experiment = Experiment.new("Sig1.0Std1in1out 2-Class classifier, 2 gaussian clusters with Sigmoid IO function for simple 1 in 1 out network", baseRandomNumberSeed)
experiment.performSimulation()


#baseRandomNumberSeed = 0
#numberOfSimulations = 1
#
#numberOfSimulations.times do |i|
#  multiRunSeed = i + baseRandomNumberSeed
#  experiment = Experiment.new("Sig0.5Std1in1out 2-Class classifier, 2 gaussian clusters with Sigmoid IO function for simple 1 in 1 out network", multiRunSeed)
#  experiment.performSimulation()
#end
#
# require_relative 'PostProcessing'


class OutputNeuron
  include NonMonotonicIOFunction
  #include PiecewiseLinNonMonIOFunction
end

baseRandomNumberSeed = 0
experiment = Experiment.new("NonM1.0Std1in1out 2-Class classifier, 2 gaussian clusters with Sigmoid IO function for simple 1 in 1 out network", baseRandomNumberSeed)
experiment.performSimulation()
