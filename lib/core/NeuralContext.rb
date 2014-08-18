### VERSION "nCore"
## ../nCore/lib/core/NeuralContext.rb

require_relative 'Utilities'
require_relative 'NeuralParts'


class NeuronInContext < Neuron
  attr_accessor :neuronControllingLearning, :layerLearningController, :adjustmentToLearningRate

  def postInitialize
    super
    @neuronControllingLearning = nil
    @layerLearningController = nil
    @adjustmentToLearningRate = nil
  end

  def calcDeltaWsAndAccumulate
    self.adjustmentToLearningRate = adjustmentTransformer(neuronControllingLearning.output, layerLearningController.output)
    inputLinks.each do |inputLink|
      inputLink.calcDeltaWAndAccumulate
    end
  end


  def adjustmentTransformer(controlNeuronsOutput, layerControllersOutput)
    adjustment = controlNeuronsOutput * layerControllersOutput
    transformedAdjustment =  if adjustment >= 0.5
                               1.0
                             else
                               0.0
                             end
    #transformedAdjustment = 1.0
  end


  def to_s
    super
    description += "\t\t\tNeuron Controlling Learning\t#{neuronControllingLearning}\n"
    description += "\t\t\tLayer Controller\t#{layerLearningController}\n"
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


class LayerLearningController

  def output
    1.0
  end

end






