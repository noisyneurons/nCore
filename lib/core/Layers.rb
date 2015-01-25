### VERSION "nCore"
; ## ../nCore/lib/core/Layers.rb
; ######################## Layer and LayerAry ######################

class Layer
  attr_reader :arrayOfNeurons
  extend Forwardable

  def initialize(input = [])
    @arrayOfNeurons = convertInputToArrayOfNeurons(input)
  end

  def_delegators :@arrayOfNeurons, :[], :size, :length, :each, :each_with_index, :collect, :all?

  def initWeights
    arrayOfNeurons.each { |aNeuron| aNeuron.initWeights }
  end

  def startStrategy
    arrayOfNeurons.each { |aNeuron| aNeuron.startStrategy }
  end

  def startEpoch
    arrayOfNeurons.each { |aNeuron| aNeuron.startEpoch }
  end

  def propagate(exampleNumber)
    arrayOfNeurons.each { |aNeuron| aNeuron.propagate(exampleNumber) }
  end

  def learnExample
    arrayOfNeurons.each { |aNeuron| aNeuron.learnExample }
  end

  def endEpoch
    arrayOfNeurons.each { |aNeuron| aNeuron.endEpoch }
  end

  def attachLearningStrategy(learningStrategy, strategyArgs)
    arrayOfNeurons.each { |aNeuron| aNeuron.learningStrat = learningStrategy.new(aNeuron, strategyArgs) }
  end

  def calcWeightsForUNNormalizedInputs
    arrayOfNeurons.each { |aNeuron| aNeuron.calcWeightsForUNNormalizedInputs }
  end

  def to_a
    return @arrayOfNeurons
  end

  def to_LayerAry
    LayerArray.new(self)
  end

  def <<(aNeuron)
    begin
      if aNeuron.kind_of?(NeuronBase)
        @arrayOfNeurons << aNeuron
        return
      end
      raise "ERROR: Attempting to append a NON-Neuron object to a Layer"
    rescue Exception => e
      logger.puts e.message
      logger.puts e.backtrace.inspect
    end
  end

  def convertInputToArrayOfNeurons(x)
    begin
      return x if (x.all? { |e| e.kind_of?(NeuronBase) }) # if x is an array of Neurons already!!
      return x if (x.length == 0) # if x is an empty array!!

      if (x.kind_of?(Array) && x.length == 1) # if x is an array of one array of Neurons already!!
        y = x[0]
        return y if (y.all? { |e| e.kind_of?(NeuronBase) })
      end

      if (x.kind_of?(LayerArray) && x.length == 1) # if x is a LayerArray with just one Layer within it!!
        return x[0].to_a
      end

      raise "Wrong Type of argument to initialize Layer: It is Not an Array of Neurons; nor an Array of an Array of Neurons; nor a Zero Length Array"
    rescue Exception => e
      logger.puts e.message
      logger.puts e.backtrace.inspect
    end
  end

  def setup?
    statusAry = arrayOfNeurons.collect { |aNeuron| aNeuron.learningStrat }
    !statusAry.include?(nil)
  end

  def to_s
    description = "Layer:\t"
    @arrayOfNeurons.each { |aNeuron| description += aNeuron.to_s }
    return description
  end
end

###############

class LayerArray
  attr_reader :arrayOfLayers
  extend Forwardable

  def initialize(arrayOfLayers=[])
    @arrayOfLayers = convertInputToArrayOfLayers(arrayOfLayers)
  end

  def_delegators :@arrayOfLayers, :[], :size, :length, :each, :collect, :include?

  def initWeights
    arrayOfLayers.each { |aLayer| aLayer.initWeights }
  end

  def startStrategy
    arrayOfLayers.each { |aLayer| aLayer.startStrategy }
  end

  def startEpoch
    arrayOfLayers.each { |aLayer| aLayer.startEpoch }
  end

  def propagate(exampleNumber)
    arrayOfLayers.each { |aLayer| aLayer.propagate(exampleNumber) }
  end

  def learnExample
    arrayOfLayers.reverse.each { |aLayer| aLayer.learnExample }
  end

  def endEpoch
    arrayOfLayers.each { |aLayer| aLayer.endEpoch }
  end

  def propagateAndLearnForAnEpoch(learningLayers, numberOfExamples)
    learningLayers.startEpoch
    numberOfExamples.times do |exampleNumber|
      propagate(exampleNumber)
      learningLayers.learnExample
    end
    learningLayers.endEpoch
  end

  def attachLearningStrategy(learningStrategy, strategyArgs)
    arrayOfLayers.each { |aLayer| aLayer.attachLearningStrategy(learningStrategy, strategyArgs) }
  end

  def calcWeightsForUNNormalizedInputs
    arrayOfLayers.each { |aLayer| aLayer.calcWeightsForUNNormalizedInputs }
  end

  def -(aLayerOraLayerArray)
    return LayerArray.new(arrayOfLayers - aLayerOraLayerArray.to_LayerAry.to_a)
  end

  def +(aLayer)
    self << aLayer
    return self
  end

  def <<(aLayer)
    begin
      if aLayer.kind_of?(Layer)
        @arrayOfLayers << aLayer
        return
      end
      raise "ERROR: Attempting to append a NON-Layer object to a LayerArray"
    rescue Exception => e
      logger.puts e.message
      logger.puts e.backtrace.inspect
    end
  end

  def to_a
    return arrayOfLayers
  end

  def to_LayerAry
    return self
  end

  def setup?
    statusAry = arrayOfLayers.collect { |aLayer| aLayer.setup? }
    !statusAry.include?(false)
  end

  def convertInputToArrayOfLayers(x) # convert x to ARRAY of Layers
    begin
      return x if (x.all? { |e| e.kind_of?(Layer) }) # if array of array of Layers
      return [] if (x.length == 0) # if empty array
      return [x] if (x.kind_of?(Layer)) # if a single Layer

      x = [x] if (x.all? { |e| e.kind_of?(NeuronBase) }) # if single array neurons, then convert to array of array of neurons
      if (x.all? { |e| e.kind_of?(Array) }) # if array of array of neurons
        if (x.flatten.all? { |e| e.kind_of?(NeuronBase) })
          return x.collect { |e| e.to_Layer }
        end
      end
      raise "Wrong Type: It is Not an Array of Layers or Neurons; nor a Zero Length Array"
    rescue Exception => e
      logger.puts e.message
      logger.puts e.backtrace.inspect
    end
  end

  def to_s
    description = "LayerArray:\n"
    @arrayOfLayers.each { |aLayer| description += aLayer.to_s }
    return description
  end

end

