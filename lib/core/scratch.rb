require 'rubygems'
require 'bundler/setup'
require_relative 'Utilities'

require '/home/mark/usr/local/ruby2.1.3/ruby/lib/ruby/2.1.0/forwardable'



class NeuronBase
  attr_accessor :id, :args, :output
  @@ID = 0

  def NeuronBase.zeroID
    @@ID = 0
  end

  def initialize(args)
    @id = @@ID
    @@ID += 1
    @args = args
    @output = 0.0
    postInitialize
  end

  def postInitialize;
  end

  ;

  def startEpoch
    puts "startEpoch in Neuron: #{id}"
  end

  def learnExample
    puts "learnExample in Neuron: #{id}"
  end

  def to_s
    description = ""
    description += "\n\t#{self.class} Class; ID = #{id}\tOutput= #{output}"
  end
end


class Layer
  attr_reader :arrayOfNeurons
  extend Forwardable

  def initialize(anArrayOfNeurons = [])
    @arrayOfNeurons = standardizeInputFormat(anArrayOfNeurons)
  end

  def_delegators :@arrayOfNeurons, :[], :size, :length, :each, :each_with_index, :collect, :all?

  def startStrategy
    arrayOfNeurons.each { |aNeuron| aNeuron.startStrategy }
  end

  def startEpoch
    arrayOfNeurons.each { |aNeuron| aNeuron.startEpoch }
  end

  def propagateExample(exampleNumber)
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
      puts e.message
      puts e.backtrace.inspect
    end
  end


  def standardizeInputFormat(x)
    begin
      return x if (x.length == 0)
      return x if (x.all? { |e| e.kind_of?(NeuronBase) })
      if (x.kind_of?(Array) && x.length == 1) # This is for the weird case where x= [[neuron1,neuron2, neuron3]]
        y = x[0]
        return y if (y.all? { |e| e.kind_of?(NeuronBase) })
      end
      raise "Wrong Type of argument to initialize Layer: It is Not an Array of Neurons; nor an Array of an Array of Neurons; nor a Zero Length Array"
    rescue Exception => e
      puts e.message
      puts e.backtrace.inspect
    end
  end

  def setup?
    statusAry = arrayOfNeurons.collect { |aNeuron| aNeuron.learningStrat }
    !statusAry.include?(nil)
  end
end

class LayerArray
  attr_reader :arrayOfLayers
  extend Forwardable

  def initialize(arrayOfLayers=[])
    @arrayOfLayers = standardizeInputFormat(arrayOfLayers)
  end

  def_delegators :@arrayOfLayers, :[], :size, :length, :each, :collect, :include?

  def startStrategy
    arrayOfLayers.each { |aLayer| aLayer.startStrategy }
  end

  def startEpoch
    arrayOfLayers.each { |aLayer| aLayer.startEpoch }
  end

  def propagateExample(exampleNumber)
    arrayOfLayers.each { |aLayer| aLayer.propagateExample(exampleNumber) }
  end

  def learnExample
    arrayOfLayers.reverse.each { |aLayer| aLayer.learnExample }
  end

  def endEpoch
    arrayOfLayers.each { |aLayer| aLayer.endEpoch }
  end

  def attachLearningStrategy(learningStrategy, strategyArgs)
    arrayOfLayers.each { |aLayer| aLayer.attachLearningStrategy(learningStrategy, strategyArgs) }
  end

  def -(aLayerOraLayerArray)
    return LayerArray.new(arrayOfLayers - stdFormat(aLayerOraLayerArray))
  end

  def +(aLayer)
    self << aLayer
    return self
  end

  def <<(aLayer)
    if aLayer.kind_of?(Layer)
      @arrayOfLayers << aLayer
    else
      STDERR.puts "ERROR: Attempting to append a NON-Layer object to a LayerArray"
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

  def standardizeInputFormat(x)
    begin
      return x if (x.length == 0)
      return x if (x.all? { |e| e.kind_of?(Layer) })
      x = [x] if (x.all? { |e| e.kind_of?(NeuronBase) })  # single array neurons to be converted to a Layer BELOW...
      if (x.all? { |e| e.kind_of?(Array) })
        if (x.flatten.all? { |e| e.kind_of?(NeuronBase) })  #  conversion to an array of Layers
          return x.collect { |e| e.to_Layer }
        end
      end
      raise "Wrong Type: It is Not an Array of Layers or Neurons; nor a Zero Length Array"
    rescue Exception => e
      puts e.message
      puts e.backtrace.inspect
    end
  end

  def stdFormat(aLayerOraLayerArray)
    begin
      return [aLayerOraLayerArray] if aLayerOraLayerArray.kind_of?(Layer)
      return aLayerOraLayerArray.to_a if aLayerOraLayerArray.kind_of?(LayerArray)
      raise "ERROR: Attempting to 'delete' a NON-Layer object from a LayerArray"
    rescue Exception => e
      puts e.message
      puts e.backtrace.inspect
    end
  end
end


n0 = NeuronBase.new({})
n1 = NeuronBase.new({})
n2 = NeuronBase.new({})
n3 = NeuronBase.new({})


aLayerOfNeurons0 = Layer.new([n0, n1])
aLayerOfNeurons1 = Layer.new
aLayerOfNeurons1 << n2
aLayerOfNeurons1 << n3

aLayerArrayX1 = LayerArray.new([aLayerOfNeurons0, aLayerOfNeurons1])

puts aLayerArrayX1.to_a
puts
aLayerArrayX1 = LayerArray.new( [ Layer.new([n1,n2]), Layer.new([n0,n3]) ])
puts aLayerArrayX1.to_a

#aLayerArray = LayerArray.new
#aLayerArray << aLayerOfNeurons0
#aLayerArray << aLayerOfNeurons1
#
#
#puts "Here 1"
#newLayerArray = (aLayerArray - aLayerOfNeurons0)
#newLayerArray.startEpoch
#
#
#puts "Here 2"
#newLayerArray2 = aLayerArray + aLayerOfNeurons0
#newLayerArray2.startEpoch
#
#
#puts "Here 3"
#newLayerArray = (newLayerArray2 - LayerArray.new(aLayerOfNeurons0))
#newLayerArray.startEpoch
#
#
#
#

