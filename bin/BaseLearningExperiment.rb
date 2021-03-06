### VERSION "nCore"
## ../nCore/bin/BaseLearningExperiments.rb


class Experiment

  attr_accessor :network, :theTrainer, :descriptionOfExperiment, :taskID, :jobID, :jobName,
                :randomNumberSeed, :dataSetGenerator, :examples, :numberOfExamples,
                :args, :trainingSequence
  attr_reader :logger
  include ExampleDistribution

  def initialize(args)
    @args = args

    @logger = @args[:logger]

    @taskID = ((ENV['TASK_ID']).to_i) || 0
    @randomNumberSeed = args[:baseRandomNumberSeed] + (taskID * 10000)
    srand(randomNumberSeed)

    @descriptionOfExperiment = args[:descriptionOfExperiment]
    @jobID = ((ENV['JOB_ID']).to_i) || 0
    @jobName = descriptionOfExperiment[0...10]

    $globalExperimentNumber = 0 # experimentLogger.experimentNumber

    @numberOfExamples = args[:numberOfExamples]
    @examples = createTrainingSet
    args[:testingExamples] = createTestingSet # TODO clumsy putting testing examples in args hash
  end

  def createTrainingSet
    classOfDataSetGenerator = args[:classOfDataSetGenerator]
    self.dataSetGenerator = classOfDataSetGenerator.new(args)
    examples = dataSetGenerator.generate(args[:numberOfExamples], args[:standardDeviationOfAddedGaussianNoise])
    logger.puts "Number of Training examples = #{examples.length}"
    logger.puts "Training Examples:"
    logger.puts examples
    return examples
  end

  def createTestingSet
    testExamples = dataSetGenerator.generate(args[:numberOfTestingExamples], args[:standardDeviationOfAddedGaussianNoise])
    logger.puts "Number of Testing examples = #{testExamples.length}"
    logger.puts "Test Examples:"
    logger.puts testExamples
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

    logger.puts "lastEpoch, trainingMSE, testMSE, startingTime, endingTime "
    logger.puts "#{lastEpoch}, #{trainingMSE}, #{testMSE}, #{startingTime}, #{endingTime}"
    return [lastEpoch, trainingMSE, testMSE, startingTime, endingTime]
  end

end


class ExperimentRunner
  attr_reader :args

  def initialize(args)
    @args = args
  end

  def repeatSimulation(numberOfReps = 1, randomSeedForSimulationSequence = @args[:baseRandomNumberSeed])
    aryOfTrainingMSEs = []
    aryOfTestingMSEs = []
    experiment = nil

    numberOfReps.times do |i|
      args[:baseRandomNumberSeed] = randomSeedForSimulationSequence + i
      experiment = Experiment.new(args)
      lastEpoch, trainingMSE, testingMSE, startingTime, endingTime = experiment.performSimulation()
      aryOfTrainingMSEs << trainingMSE
      aryOfTestingMSEs << testingMSE
    end
    args[:baseRandomNumberSeed] = randomSeedForSimulationSequence  # restored original baseRandomNumberSeed into args hash.  Not great, but...
    lastExperiment = experiment
    results = {:trainingMSEs => aryOfTrainingMSEs, :testingMSEs => aryOfTestingMSEs}
    return [lastExperiment, results]
  end
end



