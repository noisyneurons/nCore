### VERSION "nCore"
### ../nCore/test/neuralPartsExtended_test.rb

require 'test/unit'
require 'minitest/mock'
require 'minitest/reporters'
MiniTest::Reporters.use!

require_relative '../lib/core/NeuralPartsExtended'

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


def test_vectorizeEpochMeasures2
  #aNeuron = stub(:netInput => 1.0, :error => 1.5, :exampleNumber => 1)
  aNeuron = mock
  aNeuron.expects(:netInput).returns(2.0)
  aNeuron.expects(:error).returns(1.3)
  aNeuron.expects(:exampleNumber).returns(2)
  aNeuron.expects(:netInput).returns(1.0)
  aNeuron.expects(:error).returns(1.5)
  aNeuron.expects(:exampleNumber).returns(1)
  aMetricRecorder = NeuronRecorder.new(aNeuron, {})
  aMetricRecorder.recordResponsesForExample
  aMetricRecorder.recordResponsesForExample
  actual = aMetricRecorder.vectorizeEpochMeasures
  expected = [Vector[1.0, 1.5], Vector[2.0, 1.3]]
  assert_equal(expected, actual, "incorrect Vector within Array returned")
end

def test_vectorizeEpochMeasures1
  @aMetricRecorder.recordResponsesForExample
  actual = @aMetricRecorder.vectorizeEpochMeasures
  expected = [Vector[1.0, 1.5]]
  assert_equal(expected, actual, "incorrect Vector within Array returned")
end

def test_vectorizeEpochMeasures2A
  @aMetricRecorder.recordResponsesForExample
  @aMetricRecorder.recordResponsesForExample
  actual = @aMetricRecorder.vectorizeEpochMeasures
  expected = [Vector[1.0, 1.5], Vector[1.0, 1.5]]
  assert_equal(expected, actual, "incorrect Vector within Array returned")
end

