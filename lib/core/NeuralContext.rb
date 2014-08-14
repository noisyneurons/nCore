### VERSION "nCore"
## ../nCore/lib/core/NeuralContext.rb

require_relative 'Utilities'
require_relative 'NeuralParts'

class Context
  attr_accessor :enclosingNeuron, :args, :learn

  def initialize(controlledNeuron, controllingNeuron, args)
    @enclosingNeuron = enclosingNeuron
    @args = args
    @learn = true
  end

end


class NeuronLearningInContext < Neuron
  attr_accessor :controllingNeuron, :contextRequirements, :learningEnabled

  def postInitialize
    super
    @probabilityOfBeingEnabled = [args[:probabilityOfBeingEnabled], 0.01].max
    @outputWhenNeuronDisabled = self.ioFunction(0.0)
  end


  def backPropagate
    self.error = if contextRequirements.met( controllingNeuron.output )
                   super
                 else
                   0.0
                 end
  end
end


class ContextDecisionMaker



end

