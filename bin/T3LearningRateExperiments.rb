### VERSION "nCore"
## ../nCore/bin/T2LearningRateExperiments.rb
# Purpose:  To quantitatively explore the simplest clustering w/o supervision.
# This is a simplified and significantly reorganized version of 'Phase1Phase2MultiCycle.rb'

require_relative 'BaseLearningExperiment'

class Experiment
  def setParameters

    numberOfExamples = 8
    randomNumberSeed = 0

    @args = {
        :experimentNumber => ExperimentLogger.number,
        :descriptionOfExperiment => descriptionOfExperiment,
        :rng => Random.new(randomNumberSeed),

        # training parameters re. Output Error
        :outputErrorLearningRate => 0.40, #0.02,
        :minMSE => 0.0001,
        :maxNumEpochs => 4e3,

        # Network Architecture
        :numberOfInputNeurons => 2,
        :numberOfHiddenNeurons => 1,
        :numberOfOutputNeurons => 1,
        :weightRange => 1.0,
        :typeOfLink => FlockingLink,

        # Training Set parameters
        :numberOfExamples => numberOfExamples,

        # Recording and database parameters
        :intervalForSavingNeuronData => 100,
        :intervalForSavingDetailedNeuronData => 1000,
        :intervalForSavingTrainingData => 100,

        # Flocking Parameters...
        :flockingLearningRate => -0.02, # -0.04, # -0.002,
        :maxFlockingIterationsCount => 10, # 2000,
        :maxAbsFlockingErrorsPerExample => 0.002, #    0.04 / numberOfExamples = 0.005

        :typeOfClusterer => DynamicClusterer,
        :numberOfClusters => 2,
        :m => 2.0,
        :numExamples => numberOfExamples,
        :exampleVectorLength => 1,
        :delta => 1e-2,
        :maxNumberOfClusteringIterations => 10,
        :symmetricalCenters => true, # if true, speed is negatively affected

        # Inner Numeric Constraints -- used to floating point under or overflow
        :floorToPreventOverflow => 1e-30
    }
  end

  def createNetworkAndTrainer(examples)
    network = BPofFlockingNetwork.new(args)
    theTrainer = TrainingSupervisorHiddenNeuronLocalFlocking.new(examples, network, args)
    return network, theTrainer
  end
end

###################################### START of Main Learning  ##########################################

experiment = Experiment.new("T3LearningRateExperiments using correctionFactorForRateAtWhichNeuronsGainChanges")

experiment.performSimulation()
