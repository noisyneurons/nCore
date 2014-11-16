require 'rubygems'
require 'bundler/setup'

require '/home/mark/usr/local/ruby2.1.3/ruby/lib/ruby/2.1.0/forwardable'

#require_relative 'Utilities'
#require_relative 'DataSet'
#require_relative 'NeuralIOFunctions'

#
#
#require 'statsample'
#
## require '/home/mark/usr/local/ruby2.1.3/ruby/lib/ruby/gems/2.1.0/gems/statsample'
## Note R like generation of random gaussian variable
## and correlation matrix
#
#ss_analysis("Statsample::Bivariate.correlation_matrix") do
#  samples=1000
#  ds=data_frame(
#      'a'=>rnorm(samples),
#      'b'=>rnorm(samples),
#      'c'=>rnorm(samples),
#      'd'=>rnorm(samples))
#  cm=cor(ds)
#  summary(cm)
#end
#
#Statsample::Analysis.run_batch # Echo output to console

#module AddToArray
#  def testMod
#    puts "still extended: #{self.to_s}"
#  end
#
#  def -(otherArray)
#    resultantArray = super
#    resultantArray.extend(AddToArray)
#  end
#
#  def +(otherArray)
#    resultantArray = super
#    resultantArray.extend(AddToArray)
#  end
#
#  def <<(item)
#    resultantArray = super
#    resultantArray.extend(AddToArray)
#  end
#end
#
#
#a = [1,2,3]
#a.extend(AddToArray)
#a.testMod
#b = [3,4,5]
#b.extend(AddToArray)
#b.testMod
#c = a + b
#c.testMod
#c << 4
#c.testMod
#
#
##class ArrayS < Array
#  def testMod
#    puts "still extended: #{self.to_s}"
#    puts "class= #{self.class}"
#  end
#
#  def -(otherArray)
#    result = super
#    ArrayS.new(result)
#  end
#end
#
#
#a = ArrayS.new([1, 2, 3])
#a.testMod
#b = ArrayS.new([3, 4, 5])
#b.testMod
#
#d = a - b
#puts "class of d= #{d.class}"
#d.testMod

#c = ArrayS.new(d)
#c.testMod


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

  def_delegators :@arrayOfNeurons, :[], :size, :length, :all?

  def standardizeInputFormat(x)
    if (x.length == 0)
      return x
    end
    if (x.all? { |e| e.kind_of?(NeuronBase) })
      return x
    end
    STDERR.puts "Wrong Type: It is Not an Array of Neurons; nor a Zero Length Array"
  end


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

  def <<(aNeuron)
    if aNeuron.kind_of?(NeuronBase)
      @arrayOfNeurons << aNeuron
    else
      STDERR.puts "ERROR: Attempting to append an object that is NOT a neuron to a Layer; The object is #{aNeuron}"
    end
  end
end


class LayerArray
  attr_reader :arrayOfLayers
  extend Forwardable
  @arrayOfLayers = nil

  def initialize(arrayOfLayers=[])
    @arrayOfLayers = standardizeInputFormat(arrayOfLayers)
  end

  def_delegators :@arrayOfLayers, :[], :size, :length, :all?

  def standardizeInputFormat(x)
    if (x.length == 0)
      return x
    end
    if (x.all? { |e| e.kind_of?(Layer) })
      return x
    end
    if (x.all? { |e| e.kind_of?(NeuronBase) })
      return [x]
    end
    STDERR.puts "Wrong Type: It is Not an Array of Layers or Neurons; nor a Zero Length Array"
  end

  def startStrategy
    arrayOfLayers.each { |aLayer| aLayer.startStrategy }
  end

  def startEpoch
    arrayOfLayers.each { |aLayer| aLayer.startEpoch }
  end

  def propagateExample(exampleNumber)
    arrayOfLayers.each { |aLayer| aLayer.propagate(exampleNumber) }
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

  def to_a
    return arrayOfLayers
  end

  def -(aLayerOraLayerArray)
    return LayerArray.new(arrayOfLayers - stdFormat(aLayerOraLayerArray))
  end

  def stdFormat(aLayerOraLayerArray)
    return [aLayerOraLayerArray] if aLayerOraLayerArray.kind_of?(Layer)
    return aLayerOraLayerArray.to_a if aLayerOraLayerArray.kind_of?(LayerArray)
    STDERR.puts "ERROR: Attempting to 'delete' a NON-Layer object from a LayerArray; The object is #{otherArray}"
  end

  def +(aLayer)
    self << aLayer
    return self
  end

  def <<(aLayer)
    if aLayer.kind_of?(Layer)
      @arrayOfLayers << aLayer
    else
      STDERR.puts "ERROR: Attempting to append a NON-Layer object to a LayerArray; The object is #{aLayer}"
    end
  end
end

n0 = NeuronBase.new({})
n1 = NeuronBase.new({})
n2 = NeuronBase.new({})
n3 = NeuronBase.new({})


aLayerOfNeurons0 = Layer.new([n0, n1])
aLayerOfNeurons0.startEpoch
puts
aLayerOfNeurons1 = Layer.new
aLayerOfNeurons1 << n2
aLayerOfNeurons1 << n3


aLayerArray = LayerArray.new
aLayerArray << aLayerOfNeurons0
aLayerArray << aLayerOfNeurons1


puts "Here 1"
newLayerArray = (aLayerArray - aLayerOfNeurons0)
newLayerArray.startEpoch


puts "Here 2"
newLayerArray2 = aLayerArray + aLayerOfNeurons0
newLayerArray2.startEpoch


puts "Here 3"
newLayerArray = (newLayerArray2 - LayerArray.new(aLayerOfNeurons0))
newLayerArray.startEpoch





