### VERSION "nCore"
## ../nCore/lib/core/NeuralContext.rb

require_relative 'Utilities'
require_relative 'NeuralParts'


class NeuronInContext < Neuron
  attr_accessor :learningController

  def postInitialize
    super
    @learningController = DummyLearningController.new(self)
  end


  def to_s
    description = super
    description += "\t\t\t\t\tLearning Signal:\t#{learningController.class}\n"
  #                  Class; ID = #{learningController.id}\n"
    return description
  end
end


module LearningInContext

  def propagateForNormalization
    inputForThisExample = inputNeuron.output
    self.inputsOverEpoch << inputForThisExample   if(outputNeuron.learningController.output == 1.0)
    return inputForThisExample * weight
  end

  def calcDeltaWAndAccumulate
    self.deltaWAccumulated += calcDeltaW * outputNeuron.learningController.output
  end
end

#class LinkInContext < Link
#  include LearningInContext
#end


class LearningController
  attr_accessor :sensor

  def initialize(sensor)
    @sensor = sensor
  end

  def output
      1.0
  end

  def transform(input)
    if input >= 0.5
      1.0
    else
      0.0
    end
  end
end


class DummyLearningController < LearningController
  def output
    aGroupOfExampleNumbers = [0,1,2,3,8,9,10,11] # (0..7).to_a #
    if aGroupOfExampleNumbers.include?(sensor.exampleNumber)
      1.0
    else
      0.0
    end
  end
end




