### VERSION "nCore"
## ../nCore/lib/core/NeuralContext.rb

require_relative 'Utilities'
require_relative 'NeuralParts'


class NeuronInContext < Neuron
  attr_accessor :learningActivator

  def postInitialize
    super
    @learningActivator = nil
  end


  def to_s
    description = super
    description += "\t\t\t\t\tLearning Signal:\t#{
    learningActivator.class} Class; ID = #{learningActivator.id}\n"
    return description
  end
end


module LearningInContext

  def calcDeltaWAndAccumulate
    self.deltaWAccumulated += calcDeltaW * transform(outputNeuron.learningActivator.output)
  end

  def transform(input)
    if input >= 0.5
      1.0
    else
      0.0
    end
  end
end

class LinkInContext < Link
  include LearningInContext
end





