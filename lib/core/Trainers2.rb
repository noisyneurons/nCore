### VERSION "nCore"
## ../nCore/lib/core/Trainers2.rb

############################################################

class LearningStrategyBase # strategy for standard bp learning for output neurons
  attr_reader :neuron, :strategyArgs, :inputLinks, :outputLinks
  include CommonNeuronCalculations

  def initialize(theEnclosingNeuron, **strategyArgs)
    @neuron = theEnclosingNeuron
    @strategyArgs = strategyArgs
    ioFunction = @strategyArgs[:ioFunction]
    self.extend(ioFunction)
    @inputLinks = @neuron.inputLinks
    @outputLinks = @neuron.outputLinks if @neuron.respond_to?(:outputLinks)
  end

  def startStrategy
    neuron.output = ioFunction(neuron.netInput) # Probably only need to do this in special cases, like simulating recurrent nets
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
    @targetPlus = 2.5 # TODO need "exact number" here. -- just for illustration purposes...
    @targetMinus = -1.0 * @targetPlus
    @distanceBetweenTargets = @targetPlus - @targetMinus
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

##############

class AdapterForContext
  attr_accessor :theEnclosingNeuron, :strategyArgs, :learningStrat, :contextController
  include ForwardingToLearningStrategy

  def initialize(theEnclosingNeuron, strategyArgs)
    @theEnclosingNeuron = theEnclosingNeuron
    @strategyArgs = strategyArgs
    @learningStrat = @strategyArgs[:strategy].new(theEnclosingNeuron, strategyArgs)
    @contextController = @strategyArgs[:contextController]
  end

  def startEpoch
    learningStrat.startEpoch
  end

  def propagate(exampleNumber)
    if contextController.output == 1.0
      learningStrat.propagate(exampleNumber)
    else
      dropOutNeuron
    end
  end

  def learnExample
    if contextController.output == 1.0
      learningStrat.learnExample
    end
  end

  def endEpoch
    learningStrat.endEpoch
  end

  protected

  def dropOutNeuron
    theEnclosingNeuron.netInput = netInput = 0.0
    theEnclosingNeuron.output = learningStrat.ioFunction(netInput)
  end
end

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


