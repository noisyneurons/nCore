### VERSION "nCore"
## ../nCore/lib/core/Trainers2.rb


########################################################################
class LearningStrategyBase # strategy for standard bp learning for output neurons
  attr_reader :neuron, :strategyArgs, :inputLinks, :outputLinks
  include CommonNeuronCalculations

  def initialize(theEnclosingNeuron, ** strategyArgs)
    @neuron = theEnclosingNeuron
    @strategyArgs = strategyArgs
    ioFunction = @strategyArgs[:ioFunction]
    self.extend(ioFunction)
    @inputLinks = @neuron.inputLinks
    @outputLinks = @neuron.outputLinks if @neuron.respond_to?(:outputLinks)
    @targetPlus = self.findNetInputThatGeneratesMaximumOutput
    @targetMinus = -1.0 * @targetPlus
    @distanceBetweenTargets = @targetPlus - @targetMinus
  end

  def startStrategy
    ## neuron.output = ioFunction(neuron.netInput) # Probably only need to do this in special cases, like simulating recurrent nets
  end

  def finishLearningStrategy
  end

  # service routines that may be used by various learning strategies

  def calcWeightsForUNNormalizedInputs
    biasWeight = inputLinks.inject(0.0) { |sum, link| sum + link.propagateUsingZeroInput }
    inputLinks.each { |aLink| aLink.calcWeightsForUNNormalizedInputs }
    inputLinks[-1].weight = biasWeight
    inputLinks.each { |aLink| aLink.resetAllNormalizationVariables }
  end
end

class LearningBP < LearningStrategyBase # strategy for standard bp learning for hidden neurons

  def startEpoch
    zeroDeltaWAccumulated
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
  end

  def learnExample
    neuron.error = calcNetError * ioDerivativeFromNetInput(neuron.netInput)
    calcDeltaWsAndAccumulate
  end

  def endEpoch
    addAccumulationToWeight
  end
end

class LearningBPOutput < LearningStrategyBase # strategy for standard bp learning for output neurons

  def startEpoch
    zeroDeltaWAccumulated
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron()
    neuron.output = output = ioFunction(netInput)
    neuron.target = target = neuron.arrayOfSelectedData[exampleNumber]
    neuron.outputError = output - target
  end

  def learnExample
    neuron.error = neuron.outputError * ioDerivativeFromNetInput(neuron.netInput)
    calcDeltaWsAndAccumulate
  end

  def endEpoch
    addAccumulationToWeight
  end
end

class Normalization < LearningStrategyBase
  #def startStrategy
  #  super
  #  numberOfInputsToNeuron = inputLinks.length
  #  inputLinks.each do |aLink|
  #    verySmallNoise = 0.0001 * (rand - 0.5)
  #    aLink.weight = (0.2 + verySmallNoise) / numberOfInputsToNeuron
  #  end
  #end

  def startEpoch
    inputLinks.each { |aLink| aLink.resetAllNormalizationVariables }
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
  end

  def learnExample
    inputLinks.each { |link| link.storeEpochHistory }
  end

  def endEpoch
    inputLinks.each { |aLink| aLink.calculateNormalizationCoefficients }
  end
end

class SelfOrgStrat < LearningStrategyBase
  attr_accessor :targetMinus, :distanceBetweenTargets

  def startEpoch
    zeroDeltaWAccumulated
  end

  def propagate(exampleNumber)
    neuron.exampleNumber = exampleNumber
    neuron.netInput = netInput = calcNetInputToNeuron
    neuron.output = ioFunction(netInput)
  end

  def learnExample
    calcSelfOrgError
    calcDeltaWsAndAccumulate
  end

  def endEpoch
    addAccumulationToWeight
  end

  private

  def calcSelfOrgError
    netInput = neuron.netInput
    neuron.error = -1.0 * neuron.ioDerivativeFromNetInput(netInput) * (((netInput - targetMinus)/distanceBetweenTargets) - 0.5)
  end
end

###

module ContextForLearning
  attr_accessor :learningController

  def propagate(exampleNumber)
    if learningController.output == 1.0
      super(exampleNumber)
    else
      dropOutNeuron
    end
  end

  def learnExample
    if learningController.output == 1.0
      super
    end
  end

  protected

  def dropOutNeuron
    neuron.netInput = netInput = 0.0
    neuron.output = ioFunction(netInput)
  end
end

#class SelfOrgStratWithContext < SelfOrgStrat
#  include ContextForLearning
#end
#
#class NormalizationWithContext < Normalization
#  include ContextForLearning
#end
#

########################################################################

class LearningController
  attr_accessor :sensor

  def initialize(sensor=nil)
    @sensor = sensor
  end

  def output
    1.0
  end
end

class LearningControlledByNeuron < LearningController
  def output
    transform(sensor.output)
  end

  def transform(input)
    if input >= 0.5
      1.0
    else
      0.0
    end
  end
end

class LearningControlledByNeuronNOT < LearningControlledByNeuron
  def output
    logicalNOT(transform(sensor.output))
  end

  protected

  def logicalNOT(input)
    returnValue = if input == 1.0
                    0.0
                  else
                    1.0
                  end
  end
end

class DummyLearningController < LearningController
  def output
    #  aGroupOfExampleNumbers =  [0,1,2,3,8,9,10,11]  # (0..7).to_a #
    aGroupOfExampleNumbers = (0..15).to_a
    if aGroupOfExampleNumbers.include?(sensor.exampleNumber)
      1.0
    else
      0.0
    end
  end
end


