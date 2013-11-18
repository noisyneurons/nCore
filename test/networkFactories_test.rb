### VERSION "nCore"
## ../nCore/test/networkFactories_test.rb
# unit testing is done on the NetworkFactories code

require 'test/unit'
# require 'minitest/reporters'
# MiniTest::Reporters.use!

require_relative '../lib/core/NetworkFactories'

class Link # only need this class modification to test 'default' for create_link
  attr_accessor :args

  def initialize(inputNeuron, outputNeuron, args)
    @args = args
  end

  def to_s
    super
  end
end

############################################################
############################################################      N
class TestNeuronToNeuronConnection < MiniTest::Unit::TestCase
  include NeuronToNeuronConnection

  #-----------------------------------------------------------
  ##### Test Stubs and other Misc Fixtures... ################
  class DummyBaseNeuron
    attr_accessor :keyToExampleData, :arrayOfSelectedData

    def initialize(args)
      @keyToExampleData = nil
      @arrayOfSelectedData = []
    end
  end

  class DummyInputNeuron < DummyBaseNeuron
    attr_accessor :outputLinks

    def initialize(args)
      super
      @keyToExampleData = :inputs
      @outputLinks = []
    end
  end

  class DummyOutputNeuron < DummyBaseNeuron
    attr_accessor :inputLinks

    def initialize(args)
      super
      @keyToExampleData = :targets
      @inputLinks = []
    end
  end

  class DummyBaseController
    attr_accessor :associatedNeuron, :args

    def initialize(associatedNeuron, args)
      @associatedNeuron = associatedNeuron
      @args = args
    end
  end

  class DummyInputNeuronController < DummyBaseController
  end

  class DummyOutputNeuronController < DummyBaseController
  end

  class DummyLink
    attr_accessor :inputNeuron, :outputNeuron, :args

    def initialize(inputNeuron, outputNeuron, args)
      @inputNeuron = inputNeuron
      @outputNeuron = outputNeuron
      @args = args
    end

    def to_s
      super
    end
  end
  ##--------------------------------------------------------
  def setup
    srand(0)
    @args = {:testValue => 122.0}

    @theFirstInputNeuron = DummyInputNeuron.new(@args)
    @theSecondInputNeuron = DummyInputNeuron.new(@args)
    @inputLayer = [@theFirstInputNeuron, @theSecondInputNeuron]
    @aDummyInputNeuron = DummyInputNeuron.new(@args)
    @aDummyOutputNeuron = DummyOutputNeuron.new(@args)

    @sendingLayer = @inputLayer

    @theFirstOutputNeuron = DummyOutputNeuron.new(@args)
    @theSecondOutputNeuron = DummyOutputNeuron.new(@args)
    @receivingLayer = [@theFirstOutputNeuron, @theSecondOutputNeuron]
  end

  def test_createArrayOfNeurons1
    arrayOfNeurons = createArrayOfNeurons(DummyInputNeuron, numberOfNeurons=3, args={:test => 22})
    assert_instance_of(DummyInputNeuron, arrayOfNeurons[0], "wrong type of neuron created")
  end

  def test_createLink
    aLink = createLink(@aDummyInputNeuron, @aDummyOutputNeuron, @args)
    assert_instance_of(Link, aLink, "wrong type of Link created")
  end

  def test_createLink2
    @args[:typeOfLink] = DummyLink
    aLink = createLink(@aDummyInputNeuron, @aDummyOutputNeuron, @args)
    assert_instance_of(DummyLink, aLink, "wrong type of Link created")
  end

  def test_connect_layer_to_another
    @args[:typeOfLink] = DummyLink
    connect_layer_to_another(@sendingLayer, @receivingLayer, @args)

    firstLink = @theFirstInputNeuron.outputLinks[0]
    secondLink = @theFirstInputNeuron.outputLinks[1]
    firstLinkR = @theFirstOutputNeuron.inputLinks[0]
    secondLinkR = @theSecondOutputNeuron.inputLinks[0]
    thirdLink = @theSecondInputNeuron.outputLinks[0]
    fourthLink = @theSecondInputNeuron.outputLinks[1]
    thirdLinkR = @theFirstOutputNeuron.inputLinks[1]
    fourthLinkR = @theSecondOutputNeuron.inputLinks[1]

    assert_equal(firstLink, firstLinkR, "incorrect link connected between first sending and receiving neurons")
    assert_equal(secondLink, secondLinkR, "incorrect link connected between neurons")
    assert_equal(thirdLink, thirdLinkR, "incorrect link connected between neurons")
    assert_equal(fourthLink, fourthLinkR, "incorrect link connected between neurons")
  end

  def test_connect_layer_to_another2
    @args[:typeOfLink] = DummyLink
    connect_layer_to_another(@sendingLayer, @receivingLayer, @args)
    fourthLink = @theSecondInputNeuron.outputLinks[1]
    expectedInputNeuron = @theSecondInputNeuron
    actualInputNeuron = fourthLink.inputNeuron
    assert_equal(expectedInputNeuron, actualInputNeuron, "incorrect neurons associated with 4th link connected between second sending and second neurons")
  end

  def test_connect_layer_to_another3
    @args[:typeOfLink] = DummyLink
    connect_layer_to_another(@sendingLayer, @receivingLayer, @args)
    firstLink = @theFirstInputNeuron.outputLinks[0]
    expectedInputNeuron = @theFirstInputNeuron
    actualInputNeuron = firstLink.inputNeuron
    assert_equal(expectedInputNeuron, actualInputNeuron, "incorrect neurons associated with 4th link connected between second sending and second receiving neurons")
  end

  def test_connect_layer_to_another4
    @args[:typeOfLink] = DummyLink
    connect_layer_to_another(@sendingLayer, @receivingLayer, @args)
    firstLink = @theFirstInputNeuron.outputLinks[0]
    expectedOutputNeuron = @theFirstOutputNeuron
    actualOutputNeuron = firstLink.outputNeuron
    assert_equal(expectedOutputNeuron, actualOutputNeuron, "incorrect neurons associated with 1st link connected between the first sending and first receiving neurons")
  end

  def test_connect_layer_to_another5
    @args[:typeOfLink] = DummyLink
    connect_layer_to_another(@sendingLayer, @receivingLayer, @args)
    fourthLink = @theSecondInputNeuron.outputLinks[1]
    expectedOutputNeuron = @theSecondOutputNeuron
    actualOutputNeuron = fourthLink.outputNeuron
    assert_equal(expectedOutputNeuron, actualOutputNeuron, "incorrect neurons associated with 1st link connected between the first sending and first receiving neurons")
  end
end

#-----------------------------------------------------------
############################################################
##### Temp Class Modifications, Test Stubbing,  ############
##### and any other Misc Fixtures... #######################
class NeuronBase
  def NeuronBase.zeroID;
  end
end

class BiasNeuron
  attr_accessor :outputLinks

  def initialize(a)
    @outputLinks = []
  end
end

class Link
  attr_accessor :inputNeuron, :outputNeuron

  def initialize(inputNeuron, outputNeuron, args)
    @inputNeuron = inputNeuron
    @outputNeuron = outputNeuron
  end
end

class Neuron
  attr_accessor :inputLinks, :outputLinks

  def initialize(a)
    @inputLinks = []
    @outputLinks = []
  end
end

class InputNeuron < Neuron;
end

class OutputNeuron < Neuron
  def calcSumOfSquaredErrors
    return 1.11
  end
end

class LearningNetwork
  attr_accessor :createAndConnectLayer, :addLinksFromBiasNeuronToHiddenAndOutputNeurons
end
#-----------------------------------------------------------

#############################################################
#class TestLearningNetwork1 < MiniTest::Unit::TestCase
#  def setup
#    srand(0)
#    @args = {}
#    @aLearningNetwork = BaseNetwork.new(nil, @args)
#    a = nil
#    @arrayOfNeurons = [Neuron.new(a), Neuron.new(a)]
#    @aSecondArrayOfNeurons = [Neuron.new(a), Neuron.new(a), Neuron.new(a)]
#    @anArrayOfArraysOfNeurons = [@arrayOfNeurons, @aSecondArrayOfNeurons]
#    @singleArrayOfAllNeuronsToReceiveBiasInput = @anArrayOfArraysOfNeurons.flatten
#  end
#
#  def test_initialize1
#    assert_equal([], @aLearningNetwork.allNeuronLayers, "allNeuronLayers not initialized to empty array")
#  end
#
#  def test_initialize3
#    assert_instance_of(BiasNeuron, @aLearningNetwork.theBiasNeuron, "theBiasNeuron not initialized")
#  end
#
#  def test_addLinksFromBiasNeuronToHiddenAndOutputNeurons1
#    @aLearningNetwork.addLinksFromBiasNeuronToHiddenAndOutputNeurons(@singleArrayOfAllNeuronsToReceiveBiasInput)
#    expected = 5
#    actual = @aLearningNetwork.theBiasNeuron.outputLinks.length
#    assert_equal(expected, actual, "wrong number of links going out of Bias Neuron")
#  end
#
#  def test_addLinksFromBiasNeuronToHiddenAndOutputNeurons2
#    @aLearningNetwork.addLinksFromBiasNeuronToHiddenAndOutputNeurons(@singleArrayOfAllNeuronsToReceiveBiasInput)
#    expected = 1
#    actual = @singleArrayOfAllNeuronsToReceiveBiasInput[0].inputLinks.length
#    assert_equal(expected, actual, "wrong number of links going into this neuron")
#  end
#
#  def test_addLinksFromBiasNeuronToHiddenAndOutputNeurons3
#    @aLearningNetwork.addLinksFromBiasNeuronToHiddenAndOutputNeurons(@singleArrayOfAllNeuronsToReceiveBiasInput)
#    expected = 1
#    actual = @singleArrayOfAllNeuronsToReceiveBiasInput[4].inputLinks.length
#    assert_equal(expected, actual, "wrong number of links going into this neuron")
#  end
#
#  def test_addLinksFromBiasNeuronToHiddenAndOutputNeurons4
#    @aLearningNetwork.addLinksFromBiasNeuronToHiddenAndOutputNeurons(@singleArrayOfAllNeuronsToReceiveBiasInput)
#    expected = Link
#    actual = @aLearningNetwork.theBiasNeuron.outputLinks[4]
#    assert_instance_of(expected, actual, "wrong type (should be a Link) going from the Bias Neuron to the 5th neuron")
#  end
#
#  def test_addLinksFromBiasNeuronToHiddenAndOutputNeurons5
#    @aLearningNetwork.addLinksFromBiasNeuronToHiddenAndOutputNeurons(@singleArrayOfAllNeuronsToReceiveBiasInput)
#    expected = Link
#    actual = @singleArrayOfAllNeuronsToReceiveBiasInput[4].inputLinks[0]
#    assert_instance_of(expected, actual, "wrong type (should be a Link) going into the 5th neuron")
#  end
#
#  def test_createAndConnectLayer1
#    theCreatedLayer = @aLearningNetwork.createAndConnectLayer(nil, Neuron, numberOfNeurons=4)
#    expected = 4
#    actual = theCreatedLayer.length
#    assert_equal(expected, actual, "wrong number of neurons in created layer")
#  end
#
#  def test_createAndConnectLayer2
#    theCreatedLayer = @aLearningNetwork.createAndConnectLayer(nil, Neuron, numberOfNeurons=3)
#    assert_instance_of(Neuron, theCreatedLayer[2], "wrong type populating layer")
#  end
#
#  def test_createAndConnectLayer3
#    theCreatedLayer = @aLearningNetwork.createAndConnectLayer(@arrayOfNeurons, Neuron, numberOfNeurons=3)
#    expected = 2
#    actual = theCreatedLayer[2].inputLinks.length
#    assert_equal(expected, actual, "wrong number of input links going into created layer")
#  end
#
#  def test_createAndConnectLayer4
#    theCreatedLayer = @aLearningNetwork.createAndConnectLayer(@arrayOfNeurons, Neuron, numberOfNeurons=3)
#    anExpectedLink = theCreatedLayer[2].inputLinks[1]
#    assert_instance_of(Link, anExpectedLink, "wrong type populating input links to neurons populating layer")
#  end
#
#  def test_createAndConnectLayer5
#    theCreatedLayer = @aLearningNetwork.createAndConnectLayer(@arrayOfNeurons, Neuron, numberOfNeurons=3)
#    anExpectedLink = @arrayOfNeurons[1].outputLinks[1]
#    assert_instance_of(Link, anExpectedLink, "wrong type populating output links to neurons populating input layer")
#  end
#
#  def test_createAndConnectLayer6
#    theCreatedLayer = @aLearningNetwork.createAndConnectLayer(@arrayOfNeurons, Neuron, numberOfNeurons=3)
#    aLinkLeavingANeuronInTheInputLayer = @arrayOfNeurons[1].outputLinks[2]
#    aLinkGoingIntoCreatedLayer = theCreatedLayer[2].inputLinks[1]
#    #std("Link 1\n", aLinkLeavingANeuronInTheInputLayer)
#    #std("Link 2\n", aLinkGoingIntoCreatedLayer)
#    assert_equal(aLinkLeavingANeuronInTheInputLayer, aLinkGoingIntoCreatedLayer, "the link expected to connect the input layer and the created layer is not one and the same")
#  end
#end
#
#############################################################
#class TestLearningNetwork2 < MiniTest::Unit::TestCase
#  def setup
#    srand(0)
#    @args = {:learningRate => 1.0,
#             :weightRange => 1.0,
#             :numberOfInputNeurons => 2,
#             :numberOfHiddenNeurons => 3,
#             :numberOfOutputNeurons => 1,
#             :numberOfExamples => 4
#    }
#    @aLearningNetwork = BaseNetwork.new(nil, @args)
#    @allNeuronLayers = @aLearningNetwork.createSimpleLearningANN
#  end
#
#  def test_createSimpleLearningANN1
#    expected = 3
#    actual = @allNeuronLayers.length
#    assert_equal(expected, actual, "wrong number of layers in network")
#  end
#
#  def test_createSimpleLearningANN2
#    expected = 2
#    actual = @allNeuronLayers[0].length
#    assert_equal(expected, actual, "wrong number of neurons in inputLayer")
#  end
#
#  def test_createSimpleLearningANN3
#    expected = 1
#    actual = @allNeuronLayers[2].length
#    assert_equal(expected, actual, "wrong number of neurons in outputLayer")
#  end
#
#  def test_createSimpleLearningANN4a
#    expected = 4
#    actual = @allNeuronLayers[2][0].inputLinks.length
#    assert_equal(expected, actual, "wrong number of links going into outputLayer")
#  end
#
#  def test_createSimpleLearningANN4b
#    expected = 3
#    actual = @allNeuronLayers[1][2].inputLinks.length
#    assert_equal(expected, actual, "wrong number of links going into hiddenLayer")
#  end
#
#  def test_createSimpleLearningANN4c
#    expected = 0
#    actual = @allNeuronLayers[0][1].inputLinks.length
#    assert_equal(expected, actual, "wrong number of links going into inputLayer")
#  end
#
#  def test_createSimpleLearningANN5
#    expected = 1
#    actual = @allNeuronLayers[1][1].outputLinks.length
#    assert_equal(expected, actual, "wrong number of links going to outputLayer from hidden layer")
#  end
#
#  def test_calcNetworksMeanSquareError1
#    @aLearningNetwork.allNeuronLayers = [[OutputNeuron.new(nil), OutputNeuron.new(nil)]]
#    @aLearningNetwork.numberOfExamples = 2
#    expected = (1.11 + 1.11)/(2 * 2)
#    actual = @aLearningNetwork.calcNetworksMeanSquareError
#    assert_equal(expected, actual, "Mean Square Error incorrectly calculated")
#  end
#
#  def test_calcNetworksMeanSquareError2
#    @aLearningNetwork.calcNetworksMeanSquareError
#    expected = 1.11 / 4
#    actual = @aLearningNetwork.calcNetworksMeanSquareError
#    assert_equal(expected, actual, "Mean Square Error incorrectly calculated")
#  end
#end