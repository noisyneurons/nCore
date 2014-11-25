### VERSION "nCore"
## ../nCore/bin/Proj3vsProj4.rb

# Comparing Proj 3 and Proj 4's generalization performance

# General Purpose:  Start of Project 4; project to split example set to learn sub-parts, and then combine those parts/neuron-functions that
# didn't need to be separated, but instead need to be integrated to obtain better generalization.
# Ultimate goal is develop analogy processing -- where one function useful for solving one problem
# can be of use in solving another "similar-but-different" problem.  The common function(s)/neuron(s) can be thus be 'reused' -- and even potentially made
# better by improving the accuracy of the function parameters because more examples are used to learn the parameters. ala Bayes

require_relative '../lib/core/Utilities'
require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NetworkFactories'
require_relative '../lib/core/NetworkFactories2'
require_relative '../lib/core/TrainingBase'
require_relative '../lib/core/Trainers'

require_relative '../lib/plot/CorePlottingCode'
require_relative '../lib/core/SimulationDataStore'

require_relative 'BaseLearningExperiment'
########################################################################


args = {
    :experimentNumber => $globalExperimentNumber,
    :descriptionOfExperiment => "Proj4SelfOrgContextSuper; 2 in 4 out; divide then integrate",
    :baseRandomNumberSeed => 0,

    :classOfTheNetwork => Context4LayerNetwork,
    :classOfTheTrainer => Trainer4SelfOrgContextSuper,
    :classOfDataSetGenerator => Generate4ClassDataSet,

    # training parameters
    :learningRate => 0.1,
    :minMSE => 0.0,
    :epochsForSelfOrg => 300,
    :epochsForSupervisedTraining => 2400,
    :trainingSequence => TrainingSequence,

    # Network Architecture
    :numberOfInputNeurons => 2,
    :numberOfHiddenLayer1Neurons => 1,
    :numberOfHiddenLayer2Neurons => 2,
    :numberOfOutputNeurons => 4,

    # Neural Parts Specifications
    :typeOfLink => LinkWithNormalization,
    :typeOfNeuron => Neuron2,
    :typeOfLinkToOutput => LinkWithNormalization,
    :typeOfOutputNeuron => OutputNeuron2,

    :weightRange => 0.1,

    # Training Set parameters
    :numberOfExamples => (n=16),
    :numberOfTestingExamples => (16 * n),
    :standardDeviationOfAddedGaussianNoise => 0.2,
    :verticalShift => 0.0,
    :horizontalShift => 0.0,
    :angleOfClockwiseRotationOfInputData => 0.0
}


###################################### REPEATED Experiments for comparison ##########################################

numberOfRepetitions = 16
#--------------------------------------------------------------------------------------

args[:classOfTheTrainer] = Trainer3SelfOrgContextSuper
args[:descriptionOfExperiment] = "Proj3SelfOrgContextSuper; 2 in 4 out; divide but NO Integration"
runner = ExperimentRunner.new(args)
lastExperimentProj3, resultsProj3 = runner.repeatSimulation(numberOfRepetitions)


args[:classOfTheTrainer] = Trainer4SelfOrgContextSuper
args[:descriptionOfExperiment] = "Proj4SelfOrgContextSuper; 2 in 4 out; divide then integrate"
runner = ExperimentRunner.new(args)
lastExperimentProj4, resultsProj4 = runner.repeatSimulation(numberOfRepetitions)

#--------------------------------------------------------------------------------------

args[:angleOfClockwiseRotationOfInputData] = 30.0

args[:classOfTheTrainer] = Trainer3SelfOrgContextSuper
args[:descriptionOfExperiment] = "Proj3SelfOrgContextSuper; 2 in 4 out; divide but NO Integration"
runner = ExperimentRunner.new(args)
lastExperimentProj3b, resultsProj3b = runner.repeatSimulation(numberOfRepetitions)


args[:classOfTheTrainer] = Trainer4SelfOrgContextSuper
args[:descriptionOfExperiment] = "Proj4SelfOrgContextSuper; 2 in 4 out; divide then integrate"
runner = ExperimentRunner.new(args)
lastExperimentProj4b, resultsProj4b = runner.repeatSimulation(numberOfRepetitions)

#--------------------------------------------------------------------------------------

puts "\n\nNetwork's State at End of Last Experiment for Project 3:"
puts lastExperimentProj3.network
puts "\n\nNetwork's State at End of Last Experiment for Project 4:"
puts lastExperimentProj4.network

puts "\n\nNetwork's State at End of Last Experiment for Project 3b (30degrees):"
puts lastExperimentProj3b.network
puts "\n\nNetwork's State at End of Last Experiment for Project 4b (30degrees):"
puts lastExperimentProj4b.network

#--------------------------------------------------------------------------------------

puts "\n\nExperimentName    MeanTrainingMSE               MeanTestingMSE\n"
trainingMSEs, testingMSEs = resultsProj3[:trainingMSEs], resultsProj3[:testingMSEs]
puts "Proj3             #{trainingMSEs.mean}          #{testingMSEs.mean}"
trainingMSEs, testingMSEs = resultsProj4[:trainingMSEs], resultsProj4[:testingMSEs]
puts "Proj4             #{trainingMSEs.mean}          #{testingMSEs.mean}"

puts "\n\nExperimentName    MeanTrainingMSE               MeanTestingMSE\n"
trainingMSEs, testingMSEs = resultsProj3b[:trainingMSEs], resultsProj3b[:testingMSEs]
puts "Proj3b             #{trainingMSEs.mean}          #{testingMSEs.mean}"
trainingMSEs, testingMSEs = resultsProj4b[:trainingMSEs], resultsProj4b[:testingMSEs]
puts "Proj4b             #{trainingMSEs.mean}          #{testingMSEs.mean}"





