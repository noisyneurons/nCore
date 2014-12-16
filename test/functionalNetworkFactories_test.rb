### VERSION "nCore"
## ../nCore/test/functionNetworkFactories_test.rb
# This is a Functional Test requiring other neural libraries... Therefore this test should be run AFTER the
# unit testing is done on the neural libraries: NeuralParts, DataSet, Learners, NeuronToNeuronConnections, WeightedClustering

require 'test/unit'
# require 'minitest/reporters'
# MiniTest::Reporters.use!

require_relative '../lib/core/DataSet'
require_relative '../lib/core/NeuralParts'
require_relative '../lib/core/NetworkFactories'

Tolerance = 0.00001
include CommonNeuronCalculations
############################################################
class FunctionNetworkFactoriesTest < MiniTest::Unit::TestCase
  def setup
    srand(0)
    @args = {:learningRate => 1.0,
             :weightRange => 1.0,
             :numberOfInputNeurons => 2,
             :numberOfHiddenNeurons => 3,
             :numberOfOutputNeurons => 1
    }
    @aLearningNetwork = BaseNetwork.new(@args)
    @aCreatedLearningNetwork = BaseNetwork.new(@args)

    @allNeuronLayers = @aCreatedLearningNetwork.createSimpleLearningANN
  end

  def test_initialize1
    assert_equal([], @aLearningNetwork.allNeuronLayers, "allNeuronLayers not initialized to empty array")
  end

  def test_initialize3
    assert_instance_of(BiasNeuron, @aLearningNetwork.theBiasNeuron, "theBiasNeuron not initialized")
  end

  def test_createSimpleLearningANN1a
    assert_equal(3, @allNeuronLayers.length, "wrong number of layers in network")
  end

  def test_createSimpleLearningANN1b
    assert_equal(2, @allNeuronLayers[0].length, "wrong number of neurons in first layer")
  end

  def test_createSimpleLearningANN1bType
    assert_instance_of(InputNeuron, @allNeuronLayers[0][0], "wrong type of neurons in first layer")
    assert_instance_of(InputNeuron, @allNeuronLayers[0][1], "wrong type of neurons in first layer")
  end

  def test_createSimpleLearningANN1c
    assert_equal(3, @allNeuronLayers[1].length, "wrong number of neurons in second layer")
  end

  def test_createSimpleLearningANN1cType
    assert_instance_of(Neuron, @allNeuronLayers[1][0], "wrong type of neurons in 2nd layer")
    assert_instance_of(Neuron, @allNeuronLayers[1][1], "wrong type of neurons in 2nd layer")
    assert_instance_of(Neuron, @allNeuronLayers[1][2], "wrong type of neurons in 2nd layer")
  end

  def test_createSimpleLearningANN1d
    assert_equal(1, @allNeuronLayers[2].length, "wrong number of neurons in third layer")
  end

  def test_createSimpleLearningANN1dType
    assert_instance_of(OutputNeuron, @allNeuronLayers[2][0], "wrong type of neurons in 3rd layer")
  end

  def test_biasNeuronLinkage
    assert_equal(4, @aCreatedLearningNetwork.theBiasNeuron.outputLinks.length, "wrong number of links from THE BIAS NEURON")
  end

  def test_properNeuronIDs
    assert_equal(0, @aCreatedLearningNetwork.allNeuronLayers[0][0].id, "incorrect ID for neuron")
    assert_equal(3, @aCreatedLearningNetwork.allNeuronLayers[1][1].id, "incorrect ID for neuron")
    assert_equal(5, @aCreatedLearningNetwork.allNeuronLayers[2][0].id, "incorrect ID for neuron")
    assert_equal(:BiasNeuron, @aCreatedLearningNetwork.theBiasNeuron.id, "incorrect special id for 'theBiasNeuron'")
  end
end

############################################################
class FunctionNetworkFactoriesTest2 < MiniTest::Unit::TestCase
  def setup
    srand(0)
    @args = {:learningRate => 1.0,
             :weightRange => 1.0,
             :numberOfInputNeurons => 2,
             :numberOfHiddenNeurons => 3,
             :numberOfOutputNeurons => 1
    }

    @aCreatedLearningNetwork = BaseNetwork.new(nil, @args)

    @allNeuronLayers = @aCreatedLearningNetwork.createSimpleLearningANN
    @inputNeuron0 = @allNeuronLayers[0][0]
    @inputNeuron1 = @allNeuronLayers[0][1]
    @hiddenNeuron0 = @allNeuronLayers[1][0]
    @hiddenNeuron1 = @allNeuronLayers[1][1]
    @hiddenNeuron2 = @allNeuronLayers[1][2]
    @outputNeuron0 = @allNeuronLayers[2][0]
    @theBiasNeuron = @aCreatedLearningNetwork.theBiasNeuron
  end

  def test_correctLinkageBetweenNeurons1
    aLink = @inputNeuron0.outputLinks[0]
    aSecondLink = @hiddenNeuron0.inputLinks[0]
    assert_same(aLink, aSecondLink, "incorrect link between 2 neurons")
  end

  def test_correctLinkageBetweenNeurons2
    aLink = @inputNeuron0.outputLinks[2]
    aSecondLink = @hiddenNeuron2.inputLinks[0]
    assert_same(aLink, aSecondLink, "incorrect link between 2 neurons")
  end

  def test_correctLinkageBetweenNeurons3
    aLink = @inputNeuron1.outputLinks[2]
    aSecondLink = @hiddenNeuron2.inputLinks[1]
    assert_same(aLink, aSecondLink, "incorrect link between 2 neurons")
  end

  def test_correctLinkageBetweenNeurons4
    aLink = @hiddenNeuron1.outputLinks[0]
    aSecondLink = @outputNeuron0.inputLinks[1]
    assert_same(aLink, aSecondLink, "incorrect link between 2 neurons")
  end

  def test_correctLinkageBetweenNeurons5
    aLink = @hiddenNeuron2.outputLinks[0]
    aSecondLink = @outputNeuron0.inputLinks[2]
    assert_same(aLink, aSecondLink, "incorrect link between 2 neurons")
  end

  def test_correctLinkageBetweenNeurons6
    aLink = @theBiasNeuron.outputLinks[0]
    aSecondLink = @hiddenNeuron0.inputLinks[2]
    assert_same(aLink, aSecondLink, "incorrect link between 2 neurons")
  end

  def test_correctLinkageBetweenNeurons7
    aLink = @theBiasNeuron.outputLinks.last
    aSecondLink = @outputNeuron0.inputLinks[3]
    assert_same(aLink, aSecondLink, "incorrect link between 2 neurons")
  end
end

include ExampleDistribution
############################################################
class FunctionNetworkFactoriesTest3 < MiniTest::Unit::TestCase
  def setup
    srand(0)
    # add the training examples...
    examples = []
    examples << {:inputs => [0.1], :targets => [0.1], :exampleNumber => 0}
    examples << {:inputs => [0.0], :targets => [1.0], :exampleNumber => 1}
    examples << {:inputs => [1.0], :targets => [1.0], :exampleNumber => 2}
    examples << {:inputs => [0.9], :targets => [0.3], :exampleNumber => 3}

    numberOfExamples = examples.length

    @args = {:learningRate => 1.0,
             :weightRange => 1.0,
             :numberOfInputNeurons => 1,
             :numberOfHiddenNeurons => 1,
             :numberOfOutputNeurons => 1,
             :numberOfExamples => numberOfExamples
    }

    @aCreatedLearningNetwork = BaseNetwork.new(nil, @args)
    @allNeuronLayers = @aCreatedLearningNetwork.createSimpleLearningANN
    distributeDataToInputAndOutputNeurons(examples, [@allNeuronLayers.first, @allNeuronLayers.last])

    @inputNeuron0 = @allNeuronLayers[0][0]
    @hiddenNeuron0 = @allNeuronLayers[1][0]
    @outputNeuron0 = @allNeuronLayers[2][0]
    @linkToOutputNeuron0 = @outputNeuron0.inputLinks[0]
    @linkFromBiasNeuronToOutputNeuron0 = @outputNeuron0.inputLinks[1]
    @theBiasNeuron = @aCreatedLearningNetwork.theBiasNeuron
  end

  def test_propagate1
    @inputNeuron0.propagate(3)
    actual = @inputNeuron0.output
    expected = 0.9
    assert_equal(expected, actual, "wrong input registered at output of input neuron")
  end

  def test_propagate2
    @inputNeuron0.propagate(0)
    actual = @inputNeuron0.output
    expected = 0.1
    assert_equal(expected, actual, "wrong input registered at output of input neuron")
  end

  def test_propagate3
    @outputNeuron0.propagate(3)
    actual = @outputNeuron0.target
    expected = 0.3
    assert_equal(expected, actual, "wrong target registered for output neuron")
  end

  def test_propagate4
    @outputNeuron0.propagate(0)
    actual = @outputNeuron0.target
    expected = 0.1
    assert_equal(expected, actual, "wrong target registered for output neuron")
    # logger.puts @outputNeuron0
  end

  def test_propagate5
    @outputNeuron0.propagate(3)
    actual = @outputNeuron0.output
    expected = ioFunction((@hiddenNeuron0.output * @linkToOutputNeuron0.weight) + @linkFromBiasNeuronToOutputNeuron0.weight)
    assert_equal(expected, actual, "wrong output for output neuron")
  end

  def test_propagate6
    @outputNeuron0.propagate(0)
    # logger.puts @aCreatedLearningNetwork
    actual = @outputNeuron0.output
    expected = ioFunction((@hiddenNeuron0.output * @linkToOutputNeuron0.weight) + @linkFromBiasNeuronToOutputNeuron0.weight)
    assert_equal(expected, actual, "wrong output for output neuron")
  end

  def test_backPropagate1
    @outputNeuron0.propagate(3) # target = 0.3
    @outputNeuron0.backPropagate
    expectedOutputError = @outputNeuron0.output - 0.3
    expectedErrorAtInputOfNeuron = expectedOutputError * ioDerivativeFromOutput(@outputNeuron0.output)
    actual = @outputNeuron0.error
    assert_equal(expectedErrorAtInputOfNeuron, actual, "wrong error for output neuron")
  end

  def test_backPropagate2
    @outputNeuron0.propagate(3)
    @outputNeuron0.backPropagate
    @outputNeuron0.calcWeightedErrorMetricForExample
    expected = (@outputNeuron0.output - 0.3)**2.0
    actual = @outputNeuron0.weightedErrorMetric
    assert_equal(expected, actual, "wrong weightedErrorMetric for output neuron")
  end

  def test_backPropagate3
    @outputNeuron0.propagate(0)
    @outputNeuron0.backPropagate
    @outputNeuron0.calcWeightedErrorMetricForExample
    expected = (@outputNeuron0.output - 0.1)**2.0
    actual = @outputNeuron0.weightedErrorMetric
    assert_equal(expected, actual, "wrong weightedErrorMetric for output neuron")
  end
end
