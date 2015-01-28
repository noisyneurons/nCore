### VERSION "nCore"
## ../nCore/bin/SingleNeuronDemo.rb
# 313 epochs to converge

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/Layers'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/core/NeuronLearningStrategies'
require_relative '../lib/core/Trainers'

require_relative '../lib/plot/CorePlottingCode'

require_relative 'BaseLearningExperiment'

logger = StringIO.new

args = {
    :experimentNumber => $globalExperimentNumber,
    :descriptionOfExperiment => "SingleNeuronDemo OR Demo",
    :randomNumberSeed => 0,

    :classOfTheNetwork => Simplest1LayerNet,
    :classOfTheTrainer => OneNeuronSelfOrgTrainer, #TrainerBase,
    :classOfDataSetGenerator => ORCenteredDataGenerator,

    # training parameters re. Output Error
    :learningRate => -0.3,
    :minMSE => 0.0, #0.001,
    :epochsForSelfOrg => 150, #150,
    :epochsForSupervisedTraining => 1, # 600,
    :trainingSequence => TrainingSequence,

    # Network Architecture
    :numberOfInputNeurons => 2,
    :numberOfOutputNeurons => 1,
    :weightRange => 1.0,

    :typeOfLink => Link,
    :typeOfNeuron => Neuron3,
    :typeOfOutputNeuron => OutputNeuron3,


    # Training Set parameters
    :numberOfExamples => 4,
    :numberOfTestingExamples => 4,
    :standardDeviationOfAddedGaussianNoise => 0.0,
    :verticalShift => 0.0,
    :horizontalShift => 0.0,
    :angleOfClockwiseRotationOfInputData => 0.0,


    # Results and debugging information storage/access
    :logger => logger

}

###################################### REPEATED Experiments for comparison ##########################################

numberOfRepetitions = 1


runner = ExperimentRunner.new(args)
lastExperimentRun, results = runner.repeatSimulation(numberOfRepetitions)
#logger.puts lastExperimentRun.network

loggedData = logger.string

$redis.rpush("SimulationList", loggedData)

retrievedData = $redis.rpoplpush("SimulationList", "SimulationList")

puts retrievedData

numberOfExperimentsStoredInList = $redis.llen("SimulationList")

puts "\n\nnumber Of Experiments Stored In List =\t#{numberOfExperimentsStoredInList}"