### VERSION "nCore"
## ../nCore/bin/Proj3SelfOrgContextSuper.rb

# Specific Purpose for this experiment: Get SIMPLEST versions of self-org, context, AND combined with Supervised Learning,  understood and "working."
# Purpose:  Start of Project 7; project to split example set to learn sub-parts, and then combine those parts/neuron-functions that
# didn't need to be separated, but instead need to be integrated to obtain better generalization.
# Ultimate goal of project 6 is develop analogy processing -- where one function useful for solving one problem
# can be of use in solving another problem.  The common function(s)/neuron(s) can be thus be 'reused' -- and even potentially made
# better by improving the accuracy of the function parameters because more examples are used to learn the parameters.

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


class Experiment

  def setParameters
    self.numberOfExamples = 16
    @args = {
        :experimentNumber => $globalExperimentNumber,
        :descriptionOfExperiment => "Proj3SelfOrgContextSuper; 2 in 4 out; divide but do NOT INTEGRATE!",
        :randomNumberSeed => (randomNumberSeed + 0),

        :classOfTheNetwork => Context4LayerNetwork,
        :classOfTheTrainer => Trainer3SelfOrgContextSuper,
        :classOfDataSetGenerator => Generate4ClassDataSet,

        # training parameters
        :learningRate => 0.1,
        :minMSE => 0.0,
        :epochsForSelfOrg => 150,
        :epochsForSupervisedTraining => 600,
        :trainingSequence => TrainingSequence,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenLayer1Neurons => 1,
        :numberOfHiddenLayer2Neurons => 2,
        :numberOfOutputNeurons => 4,

        :weightRange => 0.1,

        :typeOfLink => LinkWithNormalization,
        :typeOfNeuron => Neuron2,
        :typeOfLinkToOutput => LinkWithNormalization,
        :typeOfOutputNeuron => OutputNeuron2,

        # Training Set parameters
        :numberOfExamples => numberOfExamples,
        :numberOfTestingExamples => 4,
        :standardDeviationOfAddedGaussianNoise => 0.0,
        :angleOfClockwiseRotationOfInputData => 0.0
    }
  end
end

####################################### START of Main Learning  ##########################################
#
#baseRandomNumberSeed = 0
#
#experiment = Experiment.new("Proj3SelfOrgContextSuper; 2 in 4 out; divide then integrate", baseRandomNumberSeed)
#
#experiment.performSimulation()
#
#puts experiment.network

###################################### START of REPEATED Experiments ##########################################

def repeatSimulation(numberOfReps = 1, randomSeedForSimulationSequence = 0)
  aryOfTrainingMSEs = []
  aryOfTestMSEs = []
  experiment = nil

  numberOfReps.times do |i|
    experimentsRandomNumberSeed = (i + randomSeedForSimulationSequence)
    experiment = Experiment.new(experimentsRandomNumberSeed)
    lastEpoch, trainingMSE, testMSE, startingTime, endingTime = experiment.performSimulation()
    aryOfTrainingMSEs << trainingMSE
    aryOfTestMSEs << testMSE
  end
  puts "\n\nmean TrainingMSE= #{aryOfTrainingMSEs.mean},\tmean TestingMSE= #{aryOfTestMSEs.mean}"
  return experiment
end

experiment = repeatSimulation
puts experiment.network

