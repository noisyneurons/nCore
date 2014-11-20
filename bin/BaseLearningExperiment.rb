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
    @randomNumberSeed = args[:baseRandomNumberSeed] + (taskID * 10000)
    srand(randomNumberSeed)

    puts "sleeping" unless ($currentHost == "localhost")
    sleep(rand * 30) unless ($currentHost == "localhost")

    @descriptionOfExperiment = args[:descriptionOfExperiment]
    @jobID = ((ENV['JOB_ID']).to_i) || 0
    @jobName = descriptionOfExperiment[0...10]

    @experimentLogger = ExperimentLogger.new(descriptionOfExperiment, jobName)
    $globalExperimentNumber = experimentLogger.experimentNumber

    @numberOfExamples = args[:numberOfExamples]
    @examples = createTrainingSet
    args[:testingExamples] = createTestingSet # TODO clumsy putting testing examples in args hash
  end

  def createTrainingSet
    classOfDataSetGenerator = args[:classOfDataSetGenerator]
    self.dataSetGenerator = classOfDataSetGenerator.new(args)
    examples = dataSetGenerator.generate(args[:numberOfExamples], args[:standardDeviationOfAddedGaussianNoise])
    puts "Number of Training examples = #{examples.length}"
    return examples
  end

  def createTestingSet
    testExamples = dataSetGenerator.generate(args[:numberOfTestingExamples], args[:standardDeviationOfAddedGaussianNoise])
    puts "Number of Testing examples = #{testExamples.length}"
    # puts "Test Examples:"
    # puts testExamples
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

end


class ExperimentRunner
  attr_reader :args
  def initialize(args)
    @args = args
  end

  def repeatSimulation(numberOfReps = 1, randomSeedForSimulationSequence = 0)
    aryOfTrainingMSEs = []
    aryOfTestingMSEs = []
    experiment = nil

    numberOfReps.times do |i|
      experimentsRandomNumberSeed = randomSeedForSimulationSequence + i
      args[:baseRandomNumberSeed] = experimentsRandomNumberSeed
      experiment = Experiment.new(args)
      lastEpoch, trainingMSE, testingMSE, startingTime, endingTime = experiment.performSimulation()
      aryOfTrainingMSEs << trainingMSE
      aryOfTestingMSEs << testingMSE
    end
    lastExperiment = experiment
    results = {:trainingMSEs => aryOfTrainingMSEs, :testingMSEs => aryOfTestingMSEs}
    return [lastExperiment, results]
  end
end



