require_relative '../lib/core/DataSet'


class TrainerSelfOrg < TrainerBase
  attr_accessor :selfOrgNeurons

  def postInitialize
    super
    self.selfOrgNeurons = allNeuronLayers[1]
  end


  def train
    distributeSetOfExamples(examples)
    phaseTrain { performStandardBackPropTraining }
    trainingSequence.startNextPhaseOfTraining
    phaseTrain { performSelfOrgTraining }
    forEachExampleDisplayInputsAndOutputs
    testMSE = calcTestingMeanSquaredErrors
    return trainingSequence.epochs, calcMeanSumSquaredErrors, testMSE
  end


  def phaseTrain
    mse = 1e100
    while ((mse >= minMSE) && trainingSequence.stillMoreEpochs)
      yield performStandardBackPropTraining()
      neuronsWithInputLinks.each { |aNeuron| aNeuron.dbStoreNeuronData }
      dbStoreTrainingData()
      trainingSequence.nextEpoch
      mse = calcMSE
    end
    return mse
  end


  def performSelfOrgTraining
    acrossExamplesAccumulateSelfOrgDeltaWs
    neuronsWithInputLinks.each { |aNeuron| aNeuron.addAccumulationToWeight }
  end


  def acrossExamplesAccumulateSelfOrgDeltaWs
    startEpoch()
    numberOfExamples.times do |exampleNumber|
      propagateAcrossEntireNetwork(exampleNumber)
      backpropagateAcrossEntireNetwork() # really only need to do this for non-self org layers  ??
      selfOrgNeurons.each { |aNeuron| aNeuron.calcSelfOrgError }
      calcWeightedErrorMetricForExample()

      neuronsWithInputLinks.each do |aNeuron|
        dataRecord = aNeuron.recordResponsesForExample
        aNeuron.calcDeltaWsAndAccumulate
        aNeuron.dbStoreDetailedData
      end
    end
  end
end
