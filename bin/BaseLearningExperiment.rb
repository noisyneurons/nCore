### VERSION "nCore"
## ../nCore/bin/BaseLearningExperiments.rb

class Experiment
  attr_accessor :network, :theTrainer, :descriptionOfExperiment, :taskID, :jobID, :jobName,
                :experimentLogger, :simulationDataStoreManager, :randomNumberSeed,
                :dataSetGenerator, :examples, :numberOfExamples, :args, :trainingSequence
  include ExampleDistribution
  include DataSetGenerators

  def initialize(args)
    @args = args

    @taskID = ((ENV['SGE_TASK_ID']).to_i) || 0
    baseRandomNumberSeed = args[:baseRandomNumberSeed]
    @randomNumberSeed = baseRandomNumberSeed + (taskID * 10000)

    @numberOfExamples = args[:numberOfExamples]
    srand(randomNumberSeed)

    puts "sleeping" unless ($currentHost == "localhost")
    sleep(rand * 30) unless ($currentHost == "localhost")

    @descriptionOfExperiment = args[:descriptionOfExperiment]
    @jobID = ((ENV['JOB_ID']).to_i) || 0
    @jobName = descriptionOfExperiment[0...10]

    @experimentLogger = ExperimentLogger.new(descriptionOfExperiment, jobName)
    $globalExperimentNumber = experimentLogger.experimentNumber

    @examples = createTrainingSet
    args[:testingExamples] = createTestingSet # TODO clumsy putting testing examples in args hash
  end

  def createTrainingSet
    classOfDataSetGenerator = args[:classOfDataSetGenerator]
    self.dataSetGenerator = classOfDataSetGenerator.new(args)
    examples = dataSetGenerator.generate(args[:numberOfExamples], args[:standardDeviationOfAddedGaussianNoise])
    puts "length of examples = #{examples.length}"
    return examples
  end

  def createTestingSet
    testExamples = dataSetGenerator.generate(args[:numberOfTestingExamples], 0.0)
    puts "Test Examples:"
    puts testExamples
    return testExamples
  end

  def createNetworkAndTrainer
    classOfTheNetwork = args[:classOfTheNetwork]
    classOfTheTrainer = args[:classOfTheTrainer]

    network = classOfTheNetwork.new(args)
    theTrainer = classOfTheTrainer.new(examples, network, args)
    return network, theTrainer
  end

  def performSimulation

######################## Create Network and Trainer ....
    self.network, self.theTrainer = createNetworkAndTrainer

###################################### perform Learning/Training  ##########################################

    startingTime = Time.now
    lastEpoch, trainingMSE, testMSE = theTrainer.train
    endingTime = Time.now

############################## reporting results....

    puts "lastEpoch, trainingMSE, testMSE, startingTime, endingTime "
    puts "#{lastEpoch}, #{trainingMSE}, #{testMSE}, #{startingTime}, #{endingTime}"
    return [lastEpoch, trainingMSE, testMSE, startingTime, endingTime]
  end

  #def plotTrainingResults(arrayOfNeuronsToPlot)
  #  generatePlotForEachNeuron(arrayOfNeuronsToPlot) if arrayOfNeuronsToPlot.present?
  #end
end


class ExperimentRunner
  attr_reader :args
  def initialize(args)
    @args = args
  end

  def repeatSimulation(numberOfReps = 1, randomSeedForSimulationSequence = 0)
    aryOfTrainingMSEs = []
    aryOfTestMSEs = []
    experiment = nil

    numberOfReps.times do |i|
      experimentsRandomNumberSeed = randomSeedForSimulationSequence + i
      args[:baseRandomNumberSeed] = experimentsRandomNumberSeed
      experiment = Experiment.new(args)
      lastEpoch, trainingMSE, testMSE, startingTime, endingTime = experiment.performSimulation()
      aryOfTrainingMSEs << trainingMSE
      aryOfTestMSEs << testMSE
    end
    puts "\n\nmean TrainingMSE= #{aryOfTrainingMSEs.mean},\tmean TestingMSE= #{aryOfTestMSEs.mean}"
    return experiment
  end
end




#experiment = repeatSimulation
#puts experiment.network



