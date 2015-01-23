; ### VERSION "nCore"
; ## ../nCore/lib/core/MiscellaneousCode.rb
; ###################################################################

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
