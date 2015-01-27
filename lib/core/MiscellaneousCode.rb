; ### VERSION "nCore"
; ## ../nCore/lib/core/MiscellaneousCode.rb
; ###################################################################





# base Gaussian model for estimating a single gaussian's parameters
# This base model is only useful as part of a mixture of models, where some
# models parameters are "more adaptable."
# A Gaussian model to incorporate into a "mixture of models" distribution model
# of a neuron's netInputs or outputs across examples
#
# Using the following iterative method for single pass variance estimation:
# FROM  Wikipedia  http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
#def online_variance(data):
#    n = 0
#mean = 0
#M2 = 0
#
#for x in data:
#  n = n + 1
#  delta = x - mean
#  mean = mean + delta/n
#  M2 = M2 + delta*(x - mean)
#
#  if (n < 2):
#      return 0
#
#  variance = M2/(n - 1)
#  return variance
#end
class GaussModelAdaptable < GaussModel

  def initialize(mean, std, prior, numberOfExamples)
    super
    @newMean = nil
    @m2 = nil
    @delta = nil
    self.initEpoch
  end

  def initEpoch
    @n = 0.0
    @newMean = 0.0
    @m2 = 0.0
  end

  def prepForRecalculatingModelsParams(inputOrOutputForExample)
    x = inputOrOutputForExample
    @n += @examplesProbability
    @delta = (x - @newMean) * @examplesProbability
    @newMean = @newMean + (@delta / @n) if (@n > 1.0e-10)
    deltaPrime = (x - @newMean) * @examplesProbability
    @m2 = @m2 + (@delta * deltaPrime)
  end

  def atEpochsEndCalculateModelParams
    @prior = @n / @numberOfExamples
    @mean = @newMean
    variance = @m2/(@n - 1)
    newStd = Math.sqrt(variance)
    @std = newStd
  end

  def to_s
    "\t\tmean=\t#{mean}\tstd=\t#{std}\tprior=\t#{prior}\n"
  end
end

class WeightChangeNormalizer
  attr_accessor :layer, :weightChangeSetPoint

  def initialize(layer, args, weightChangeSetPoint = 0.08)
    @layer = layer
    @weightChangeSetPoint = weightChangeSetPoint
  end

  def normalizeWeightChanges
    layerGain = weightChangeSetPoint / maxWeightChange
    layer.each { |neuron| neuron.inputLinks.each { |aLink| aLink.deltaWAccumulated = layerGain * aLink.deltaWAccumulated } }
  end

  private

  def maxWeightChange
    acrossLayerMaxValues = layer.collect do |aNeuron|
      accumulatedDeltaWsAbs = aNeuron.inputLinks.collect { |aLink| aLink.deltaWAccumulated.abs }
      accumulatedDeltaWsAbs.max
    end
    return acrossLayerMaxValues.max
  end
end
; ###################################################################
class NoisyNeuron < Neuron
  attr_accessor :probabilityOfBeingEnabled, :enabled, :outputWhenNeuronDisabled, :learning

  def postInitialize
    super
    @learning = true
    @probabilityOfBeingEnabled = [args[:probabilityOfBeingEnabled], 0.01].max
    @outputWhenNeuronDisabled = self.ioFunction(0.0)
  end

  def propagate(exampleNumber)
    case
      when learning == true
        self.output = if (self.enabled = rand < probabilityOfBeingEnabled)
                        super(exampleNumber)
                      else
                        outputWhenNeuronDisabled
                      end

      when learning == false
        signal = super(exampleNumber) - outputWhenNeuronDisabled
        self.output = (signal / probabilityOfBeingEnabled) + outputWhenNeuronDisabled
      else
        logger.puts "error, 'learning' variable not set to true or false!!"
    end
    output
  end

  def backPropagate
    self.error = if (enabled)
                   super
                 else
                   0.0
                 end
  end
end

