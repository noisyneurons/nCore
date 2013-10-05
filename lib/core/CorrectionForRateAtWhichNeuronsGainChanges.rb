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
# For netInputs above 3.0 the average ratio of gains is relatively large but the average
# ratio of gains does NOT change any more!  See function implementing this equation below:  "Rate of Change of the Ratio of 2 typical Neural-Gains"


require_relative 'Utilities'

module CommonClusteringCode
  #def clustersCenter
  #  clusters[0].center[0] # assuming symmetrical cluster centers therefore d output/d input of cluster[0].center or cluster[1].center are the SAME value
  #end

  def clustersCenter     # TODO This is a simple approximation IF we want to deal with non-symmetric clusters... Do we want to deal with non-symmetric clusters
  ( (clusters[0].center[0]).abs + (clusters[1].center[0]).abs ) / 2.0
  end
end


class AbstractStepTrainer
  include Math

  def adaptToLocalFlockError
    STDERR.puts "Generating neurons and adapting neurons are not one in the same.  This is NOT local flocking!!" if (flockErrorGeneratingNeurons != flockErrorAdaptingNeurons)
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.learningRate = args[:flockingLearningRate] }
    flockErrorGeneratingNeurons.each { |aNeuron| aNeuron.accumulatedAbsoluteFlockingError = 0.0 }
    acrossExamplesAccumulateFlockingErrorDeltaWs
    flockErrorAdaptingNeurons.each { |aNeuron| aNeuron.addAccumulationToWeight }
    self.accumulatedAbsoluteFlockingErrors = flockErrorGeneratingNeurons.collect { |aNeuron| (aNeuron.accumulatedAbsoluteFlockingError * correctionFactorForRateAtWhichNeuronsGainChanges(aNeuron.clustersCenter)) }
  end

  #  This method corrects for: "how the Ratio of Neural-Gains changes with the magnitudes of the example netInputs"
  def correctionFactorForRateAtWhichNeuronsGainChanges(clustersCenter, correctionFactorsFloor = 0.01)
    c = clustersCenter
    m = (exp(-c) + 1)

    n = m**2
    o = (2.0 * exp(-2.0 * c)) / (m**3)
    p = -1.0 * (exp(-c) / m**2)
    q = exp(c)

    result = (n * (o + p) * q).abs
    return correctionFactorsFloor if(result.nan?)
    return [result, correctionFactorsFloor].max
  end
end
