### VERSION "nCore"
## ../nCore/lib/core/NeuralContext.rb

require_relative 'Utilities'
require_relative 'NeuralParts'


class NeuronInContext < Neuron
  attr_accessor :neuronControllingLearning, :flipLearningProbability, :adjustmentToLearningRate

  def postInitialize
    super
    @adjustmentToLearningRate = nil
    @flipLearningProbability = false
  end

  def calcDeltaWsAndAccumulate
    self.adjustmentToLearningRate = adjustmentTransformer(neuronControllingLearning.output, flipLearningProbability)
    #if adjustmentToLearningRate > 0.0
      inputLinks.each do |inputLink|
        inputLink.calcDeltaWAndAccumulate
      end
    #end
  end

  def to_s
    description = super
    description += "\t\t\t\t\tNeuron Controlling Learning:\t#{
    neuronControllingLearning.class} Class; ID = #{neuronControllingLearning.id}\n"
    return description
  end

  protected

  def adjustmentTransformer(controlNeuronsOutput, flipLearningProbability)
    adjustment = if controlNeuronsOutput >= 0.5
                   1.0
                 else
                   0.0
                 end
    if flipLearningProbability
      1.0 - adjustment
    else
      adjustment
    end
  end

  public
end


class LinkInContext < Link

  def calcDeltaWAndAccumulate
    self.deltaWAccumulated += calcDeltaW * outputNeuron.adjustmentToLearningRate
  end
end





