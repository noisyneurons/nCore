### VERSION "nCore"
## ../nCore/lib/core/CorrectionForRateAtWhichNeuronsGainChanges.rb

# Implements a method that corrects for: "how the Ratio of Neural-Gains changes with the magnitudes of the example netInputs"

# This code modifies the Trainer.rb and NeuralPartsExtended.rb code in order to incorporate a
# correction or 'normalization' for the rate at which neural gain changes as learning progresses. I.E. The neural gains
# are different because the netInputs are different.
# In the beginning of training, the 'neural gains' for different examples
# will be similar because all the netInputs are not far from a 0.0 netInput. -- because in this netInput region the neural
# gains change with very little as the netInput is changed.
#
# However, by the time the average netInputs become larger (as they often will as learning progresses) the ratio between any
# of two of the 'neural gains' can become quite large, e.g. for netInputs > 1.0 - 2.0.
# For netInputs above 3.0 the average ratio of gains is relatively large but does the average
# ration does NOT change!  See function implementing this equation below:  "Rate of Change of the Ratio of 2 typical Neural-Gains"


require_relative 'Utilities'

module CommonClusteringCode
  def clustersCenter
    clusters[0].center
  end
end


class SimpleAdjustableLearningRateTrainer
  include Math

  def accumulateFlockingErrorDeltaWs
    adaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:flockingLearningRate] }
    adaptingNeurons.each { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError = 0.0 }
    # print "Net Input, Flocking Error=\t"
    acrossExamplesAccumulateDeltaWs do |aNeuron, dataRecord, exampleNumber|
      dataRecord[:localFlockingError] = aNeuron.calcLocalFlockingError { aNeuron.weightedExamplesCenter } if (useFuzzyClusters?)
      dataRecord[:localFlockingError] = aNeuron.calcLocalFlockingError { aNeuron.centerOfDominantClusterForExample } unless (useFuzzyClusters?)
      epochs = args[:trainingSequence].epochs
      DetailedNeuronData.new(args[:trainingSequence].epochs, aNeuron.id, exampleNumber, dataRecord[:netInput], dataRecord[:localFlockingError]) if(epochs % 100 == 0)
      # print "#{dataRecord[:netInput]},\t#{dataRecord[:localFlockingError]};\t"
      aNeuron.calcAccumDeltaWsForLocalFlocking
    end
    # puts
    adaptingNeurons.collect { |aNeuron| (aNeuron.accumulatedAbsoluteFlockingError * correctionFactorForRateAtWhichNeuronsGainChanges(aNeuron.clustersCenter)) }
  end


  # Function:
  #  Implements a method that corrects for: "how the Ratio of Neural-Gains changes with the magnitudes of the example netInputs"
  def correctionFactorForRateAtWhichNeuronsGainChanges(clustersCenter, correctionFactorsFloor = 0.1)
    c = clustersCenter[0]
    m = (exp(-c) + 1)

    n = m**2
    o = (2.0 * exp(-2.0 * c)) / (m**3)
    p = -1.0 * (exp(-c) / m**2)
    q = exp(c)

    result = (n * (o + p) * q).abs
    return [result, correctionFactorsFloor].max
  end
end
