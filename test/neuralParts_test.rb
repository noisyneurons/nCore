### VERSION "nCore"
### ../nCore/test/neuralParts_test.rb

require 'test/unit'
require 'mocha/setup'
require_relative '../lib/core/NeuralParts'

Tolerance = 0.00001

############################################################
##### Temp Class Modifications, Test Stubbing,  ############
##### and any other Misc Fixtures... #######################

class NeuronBase
  def postInitialize;
  end # ignore error message that is normally called.
end

class OutputNeuron
  public :calcWeightedErrorMetricForExample
  attr_accessor :exampleNumber
end
############################################################
############################################################

############################################################
class TestBaseNeuron < MiniTest::Unit::TestCase
  def setup
    @args = {:specification => nil}
    @neuronBase = NeuronBase.new(@args)
  end

  def test_CreateBaseNeuron
    assert_instance_of(NeuronBase, @neuronBase, "wrong class created")
  end

  def test_NeuronID
    NeuronBase.zeroID
    neuronBase0 = NeuronBase.new(@args)
    neuronBase1 = NeuronBase.new(@args)
    assert_equal(0, neuronBase0.id, "incorrect object id")
    assert_equal(1, neuronBase1.id, "incorrect object id")
  end
end

############################################################
class TestInputNeuron < MiniTest::Unit::TestCase
  def setup
    @args = {}
    @inputNeuron = InputNeuron.new(@args)
    @inputNeuron.arrayOfSelectedData = [-1.0, -0.5, 0.0]
  end

  def test_CreateInputNeuron1
    assert_instance_of(InputNeuron, @inputNeuron, "wrong class created")
  end

  def test_CreateInputNeuron3
    assert_equal([-1.0, -0.5, 0.0], @inputNeuron.arrayOfSelectedData, "array of Selected Inputs is not correct")
  end

  def test_CreateInputNeuron4
    assert_equal([], @inputNeuron.outputLinks, "There are no Output Links in array")
  end

  def test_propagate1
    exampleNumber = 0
    @inputNeuron.propagate(exampleNumber)
    assert_equal(-1.0, @inputNeuron.output, "output of the input neuron is incorrect")
  end

  def test_propagate2
    exampleNumber = 1
    @inputNeuron.propagate(exampleNumber)
    assert_equal(-0.5, @inputNeuron.output, "output of the input neuron is incorrect")
  end

  def test_to_s
    logger.puts @inputNeuron.to_s
  end
end

############################################################
class TestOutputNeuron < MiniTest::Unit::TestCase

  class DummyLink
    def propagate
      return 3.3
    end
  end

  class NeuronRecorder
    attr_accessor :withinEpochMeasures

    def initialize(a, b)
    end
  end

  def setup
    @args = {}
    @outputNeuron = OutputNeuron.new(@args)
    @outputNeuron.metricRecorder.withinEpochMeasures = [{:weightedErrorMetric => 2.37}, {:weightedErrorMetric => 1.17}]
    @outputNeuron.arrayOfSelectedData = [-1.0, -0.5, 0.3]
  end

  def test_CreateOutputNeuron1
    assert_instance_of(OutputNeuron, @outputNeuron, "wrong class created")
  end

  def test_CreateOutputNeuron3
    assert_equal([-1.0, -0.5, 0.3], @outputNeuron.arrayOfSelectedData, "array of Selected Inputs is not correct")
  end

  def test_CreateOutputNeuron4
    assert_equal([], @outputNeuron.inputLinks, "There are no input links in array")
  end

  def test_propagate1
    exampleNumber = 2
    @outputNeuron.propagate(exampleNumber)
    assert_equal(exampleNumber, @outputNeuron.exampleNumber, "example number for the output neuron is incorrect")
  end

  def test_propagate2a
    @outputNeuron.propagate(exampleNumber = 2)
    assert_equal(0.3, @outputNeuron.target, "value of target is not correct for output neuron, and example number, is incorrect")
  end

  def test_propagate2b
    @outputNeuron.inputLinks= [DummyLink.new, DummyLink.new]
    @outputNeuron.propagate(0)
    assert_equal(6.6, @outputNeuron.netInput, "netInput into the output neuron is incorrect")
  end

  def test_propagate3
    @outputNeuron.inputLinks= [DummyLink.new, DummyLink.new]
    @outputNeuron.propagate(0)
    expected = @outputNeuron.ioFunction(6.6)
    # std("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX    expected output ",expected)
    assert_equal(expected, @outputNeuron.output, "output of the output neuron is incorrect")
  end

  def test_backPropagate1

    @outputNeuron.outputError = 0.5 # and therefore outputError = 0.5 and ioDerivative = 0.8(1-0.8)= 0.16
    @outputNeuron.netInput = 0.0
    @outputNeuron.backPropagate # therefore error should be 0.08
    assert_in_delta(expected=0.125, @outputNeuron.error, (Tolerance * expected).abs, "Assuming Standard Sigmoid IO Function!! -- Calculated error is not correct for output neuron")
  end

  def test_backPropagate2
    @outputNeuron.inputLinks= [DummyLink.new, DummyLink.new] # netInput should be 6.6 and output should be ioFunction(6.6)
    @outputNeuron.propagate(2)
    expectedOutputError = @outputNeuron.ioFunction(6.6) - 0.3
    expectedError = expectedOutputError * @outputNeuron.ioDerivativeFromNetInput(6.6)
    @outputNeuron.backPropagate # therefore error should be 0.08
    assert_in_delta(expectedError, @outputNeuron.error, (Tolerance * expectedError).abs, "calculated error is not correct for output neuron")
  end

  def test_calcWeightedErrorMetricForExample
    @outputNeuron.outputError = 0.6
    @outputNeuron.calcWeightedErrorMetricForExample
    expected = 0.36
    actual = @outputNeuron.weightedErrorMetric
    assert_equal(expected, actual, "calculated error metric is not correct for output, and example number, is incorrect")
  end

  def test_calcSumOfSquaredErrors
    expected = 2.37 + 1.17
    actual = @outputNeuron.calcSumOfSquaredErrors
    assert_equal(expected, actual, "calculated Sum Of Squared Errors is not correct for output neuron")
  end

  def test_to_s
    logger.puts @outputNeuron.to_s
  end
end

############################################################
class TestBiasNeuron < MiniTest::Unit::TestCase
  def setup
    @aBiasNeuron = BiasNeuron.new(@args)
  end

  def test_BiasNeuronsOutput
    assert_equal(1.0, @aBiasNeuron.output, "incorrect output from bias neuron")
  end

  def test_to_s
    logger.puts @aBiasNeuron.to_s
  end
end

###########################################################
module PropagateInterfaceTest
  def test_implementsThePropagateInterface
    assert_respond_to(@object, :propagate)
  end
end
############################################################
module BackPropagateInterfaceTest
  def test_implementsTheBackPropagateInterface
    assert_respond_to(@object, :backPropagate)
  end
end
############################################################

############################################################
class TestDummyLink < MiniTest::Unit::TestCase

  class DummyLink
    def propagate
      return 0.3
    end

    def backPropagate
      return 0.61
    end
  end

  include PropagateInterfaceTest
  include BackPropagateInterfaceTest

  def setup
    @object = DummyLink.new
  end
end

############################################################
class TestNeuron < MiniTest::Unit::TestCase

  class DummyLink
    def propagate
      return 0.3
    end

    def backPropagate
      return 0.61
    end
  end

  include CommonNeuronCalculations
  include PropagateInterfaceTest
  include BackPropagateInterfaceTest

  def setup
    @args = {}
    @neuron = @object = Neuron.new(@args)
    @neuron2 = Neuron.new(@args)
    @mockLink1 = mock()
    @mockLink2 = mock()
    @neuron2.inputLinks << @mockLink1 << @mockLink2
  end

  def test_initialization
    assert_equal(0.0, @neuron.netInput, "wrong iniitialization of netInput")
    assert_equal(Array.new, @neuron.inputLinks, "wrong iniitialization of inputLinks")
    assert_equal(0.0, @neuron.error, "wrong iniitialization of error")
    assert_equal(self.ioFunction(0.0), @neuron.output, "wrong iniitialization of output")
  end

  def test_propagate
    @neuron.inputLinks << DummyLink.new << DummyLink.new
    assert_equal(self.ioFunction(0.6), @neuron.propagate(nil), "incorrect computation in propagate method")
  end

  def test_propagate2
    @neuron.inputLinks << DummyLink.new << DummyLink.new
    @neuron.propagate(nil)
    assert_equal(0.6, @neuron.netInput, "incorrect computation in propagate method")
  end

  def test_backPropagate
    @neuron.outputLinks.clear
    @neuron.outputLinks << DummyLink.new << DummyLink.new << DummyLink.new
    assert_equal((3.0*0.61) * self.ioDerivativeFromNetInput(netInput = 0.0), @neuron.backPropagate, "incorrect computation in backPropagate method")
  end

  def test_calcDeltaWsAndAccumulate
    @mockLink1.expects(:calcDeltaWAndAccumulate).returns(true)
    @mockLink2.expects(:calcDeltaWAndAccumulate).returns(true)
    @neuron2.calcDeltaWsAndAccumulate
  end

  def test_addAccumulationToWeight
    @mockLink1.expects(:addAccumulationToWeight).returns(true)
    @mockLink2.expects(:addAccumulationToWeight).returns(true)
    @neuron2.addAccumulationToWeight
  end

  def test_to_s
    @neuron.inputLinks << DummyLink.new << DummyLink.new
    logger.puts @neuron.to_s
  end
end

############################################################
##############################################################
class TestMetricRecorder < MiniTest::Unit::TestCase
  def setup
    @neuron = stub(:id => 111, :netInput => 1.0, :error => 1.5, :exampleNumber => 1)
    @aMetricRecorder = NeuronRecorder.new(@neuron, {})
  end

  def test_initialize1
    aSecondNeuron = rand
    aSecondMetricRecorder = NeuronRecorder.new(aSecondNeuron, {})
    assert_equal(aSecondNeuron, aSecondMetricRecorder.neuron, "neuron not embedded appropriately during initialize")
  end

  def test_initialize2
    assert_equal([], @aMetricRecorder.withinEpochMeasures, "withinEpochMeasures not initialized")
  end

  def test_recordResponsesForExample1
    @aMetricRecorder.recordResponsesForExample
    actual = @aMetricRecorder.withinEpochMeasures.length
    expected = 1
    assert_equal(expected, actual, "wrong number of items stored in 'withinEpochMeasures'")
  end

  def test_recordResponsesForExample2
    @aMetricRecorder.recordResponsesForExample
    @aMetricRecorder.recordResponsesForExample
    actual = @aMetricRecorder.withinEpochMeasures.length
    expected = 2
    assert_equal(expected, actual, "wrong number of items stored in 'withinEpochMeasures'")
  end

  def test_recordResponsesForExample3
    @aMetricRecorder.recordResponsesForExample
    @aMetricRecorder.recordResponsesForExample
    actual = @aMetricRecorder.withinEpochMeasures[1][:netInput]
    expected = 1.0
    assert_equal(expected, actual, "wrong item stored in 'withinEpochMeasures'")
  end

  def test_clearWithinBatchRecords
    @aMetricRecorder.clearWithinEpochMeasures
    storedInfo = @aMetricRecorder.withinEpochMeasures
    assert_equal([], storedInfo, "withinEpochMeasures not cleared (i.e. reset)")
  end
end

############################################################
######################## Test of Links   ####################
############################################################
############################################################
class TestLink < MiniTest::Unit::TestCase
  class DummyNeuron
    def output
      0.68
    end

    def error
      0.31
    end
  end

  def setup
    @inputNeuron = DummyNeuron.new
    @outputNeuron = DummyNeuron.new
    @aLink = Link.new(@inputNeuron, @outputNeuron,
                      {:weightRange => 1.0, :learningRate => 1.0})
  end

  def testPropagate
    expected = @inputNeuron.output * @aLink.weight
    delta = expected.abs * Tolerance
    assert_in_delta(expected, @aLink.propagate, delta, "forward propagation faulty!")
  end

  def testBackPropagate
    expected = @outputNeuron.error * @aLink.weight
    delta = expected.abs * Tolerance
    assert_in_delta(expected, @aLink.backPropagate, delta, "backward propagation faulty!")
  end

  def testCalcDeltaW
    @aLink.calcDeltaW
    expected = @aLink.learningRate * @outputNeuron.error * @inputNeuron.output
    assert_in_delta(expected, @aLink.deltaW, expected * Tolerance, "DeltaW Calculation faulty!")
  end

  def testWeightUpdate
    @aLink.weight=0.66
    @aLink.weightUpdate
    expected = @aLink.weight + @aLink.deltaW
    assert_in_delta(expected, @aLink.weight, expected * Tolerance, "Weight Update Calculation faulty!")
  end
end

