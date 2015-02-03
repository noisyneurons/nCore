### VERSION "nCore"
## ../nCore/bin/Proj3MixtureSelfOrgContextSuper.rb

# Specific Purpose for this experiment: Get SIMPLEST versions of self-org, context, AND combined with Supervised Learning,  understood and "working."
# Purpose:  Start of Project 7; project to split example set to learn sub-parts, and then combine those parts/neuron-functions that
# didn't need to be separated, but instead need to be integrated to obtain better generalization.
# Ultimate goal of project 6 is develop analogy processing -- where one function useful for solving one problem
# can be of use in solving another problem.  The common function(s)/neuron(s) can be thus be 'reused' -- and even potentially made
# better by improving the accuracy of the function parameters because more examples are used to learn the parameters.

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/Layers'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/core/NetworkFactories2'
require_relative '../lib/core/NeuronLearningStrategies'
require_relative '../lib/core/Trainers'

require_relative '../lib/plot/CorePlottingCode'

require_relative 'BaseLearningExperiment'
########################################################################

logger = StringIO.new

args = {
    :experimentNumber => $globalExperimentNumber,
    :descriptionOfExperiment => "Proj3MixtureSelfOrgContextSuper; 2 in 4 out; divide but NO Integration",
    :baseRandomNumberSeed => 0,

    :classOfTheNetwork => Context4LayerNetworkVer2,
    :classOfTheTrainer => MixtureTrainer3SelfOrgContextSuper,
    :classOfDataSetGenerator => Generate4ClassDataSet,

    # training parameters
    :learningRate => 0.1,
    :minMSE => 0.0,
    :epochsForSelfOrg => 30, # 300, for 30 degree angle rotation of data
    :epochsForSupervisedTraining => 100,
    :trainingSequence => TrainingSequence,

    # Network Architecture
    :numberOfInputNeurons => 2,
    :numberOfHiddenLayer1Neurons => 1,
    :numberOfHiddenLayer2Neurons => 2,
    :numberOfOutputNeurons => 4,

    # Neural Parts Specifications
    :typeOfLink => Link,
    :typeOfNeuron => Neuron3,
    :typeOfLinkToOutput => Link,
    :typeOfOutputNeuron => OutputNeuron3,

    :weightRange => 1.0,

    # Training Set parameters
    :numberOfExamples => 16,
    :numberOfTestingExamples => 160,
    :standardDeviationOfAddedGaussianNoise => 0.0, #0.000001,
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
$redis.save

retrievedData = $redis.rpoplpush("SimulationList", "SimulationList")

puts retrievedData

numberOfExperimentsStoredInList = $redis.llen("SimulationList")

puts "\n\nnumber Of Experiments Stored In List =\t#{numberOfExperimentsStoredInList}"