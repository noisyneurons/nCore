### VERSION "nCore"
## ../nCore/lib/core/NeuralAdaptiveMap.rb

require_relative 'Utilities'
require_relative 'NeuralParts'

### temp comment

class NeuronSOM < Neuron
  attr_accessor :relevance

  def propagate(exampleNumber)
    super(exampleNumber)
    self.relevance = ioDerivativeFromOutput(output)
  end
end



