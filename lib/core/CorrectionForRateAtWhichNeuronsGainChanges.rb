### VERSION "nCore"
## ../nCore/lib/core/CorrectionForRateAtWhichNeuronsGainChanges.rb
# This code modifies the Trainer.rb and NeuralPartsExtended.rb code in order to incorporate a
# correction or 'normalization' or the rate at which neural gain changes as learning progresses.  In the
# beginning of training, the 'neural gains' for different examples (with correspondingly different netInputs)
# will all be very similar because all the netInputs are not far from netInput = 0.  However, by the time
# in learning that the netInputs become larger than 1.0 - 3.0, the ratio between any of two of the 'neural gains'
# can become quite large.  Above 3.0 the ratio-differences in these neural gains do NOT increase much at all.


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
    acrossExamplesAccumulateDeltaWs do |aNeuron, dataRecord|
      dataRecord[:localFlockingError] = aNeuron.calcLocalFlockingError { aNeuron.weightedExamplesCenter } if (useFuzzyClusters?)
      dataRecord[:localFlockingError] = aNeuron.calcLocalFlockingError { aNeuron.centerOfDominantClusterForExample } unless (useFuzzyClusters?)
      aNeuron.calcAccumDeltaWsForLocalFlocking
    end
    adaptingNeurons.collect { |aNeuron| (aNeuron.accumulatedAbsoluteFlockingError * correctionFactorForRateAtWhichNeuronsGainChanges(aNeuron.clustersCenter)) }
  end

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
