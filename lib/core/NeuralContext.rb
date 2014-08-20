### VERSION "nCore"
## ../nCore/lib/core/NeuralContext.rb

require_relative 'Utilities'
require_relative 'NeuralParts'


class NeuronInContext < Neuron
  attr_accessor :neuronControllingLearning, :adjustmentToLearningRate

  def postInitialize
    super
    @adjustmentToLearningRate = nil
  end

  def calcDeltaWsAndAccumulate
    self.adjustmentToLearningRate = adjustmentTransformer(neuronControllingLearning.output)
    inputLinks.each do |inputLink|
      inputLink.calcDeltaWAndAccumulate
    end
  end


  def adjustmentTransformer(controlNeuronsOutput)
    transformedAdjustment =  if controlNeuronsOutput >= 0.5
                               1.0
                             else
                               0.0
                             end
    #transformedAdjustment = 1.0
  end


  def to_s
    description = super
    description += "\t\t\tNeuron Controlling Learning\t#{neuronControllingLearning}\n"
    return description
  end
end


class LinkInContext < Link
  def calcDeltaWAndAccumulate
    adjustmentToLearningRate = outputNeuron.adjustmentToLearningRate
    self.deltaWAccumulated += if adjustmentToLearningRate > 0.0
                                adjustmentToLearningRate * calcDeltaW
                              else
                                0.0
                              end
  end
end


#class LayerLearningController
#
#  def output
#    1.0
#  end
#
#end
#





