### VERSION "nCore"
## ../nCore/test/TestFunctionOfSelfOrgLearnerSystem.rb
# Test of Flocking/Clustering SelfOrgLearnerSystem
# This is a Functional Test requiring other neural libraries... Therefore this test should be run AFTER the
# unit testing is done on the neural libraries: NeuralParts, NeuralPartsExtended, DataSet, NeuronToNeuronConnections, WeightedClustering

require 'test/unit'
require 'minitest/unit'
require 'minitest/mock'

require_relative '../lib/core/NeuralPartsExtended'

Tolerance = 0.00001
############################################################      N
class TestOfFlockingNeuronAndDynamicClusterer < MiniTest::Unit::TestCase
  def setup
    srand(0)
    @testData = [{:netInput => -1.0, :error => -1.0}, {:netInput => -1.0, :error => -1.0},
                 {:netInput => 1.0, :error => 1.0}, {:netInput => 1.0, :error => 1.0}]

    @vectorizedTestData = self.convertEachHashToAVector(@testData)

    args = {:typeOfClusterer => DynamicClusterer,
            :numberOfClusters => 2,
            :m => 2.0,
            :numExamples => 4,
            :exampleVectorLength => 2,
            :delta => 0.001,
            :maxNumberOfClusteringIterations => 100
    }

    @aFlockingNeuron = FlockingNeuron.new(args)

    @metricRecorder = MiniTest::Mock.new
    @aFlockingNeuron.metricRecorder= @metricRecorder

    @clusters = @aFlockingNeuron.clusterer.clusters
  end

  def convertEachHashToAVector(anArray)
    anArrayOfVectors = anArray.collect do |measuresForAnExample|
      Vector[(measuresForAnExample[:netInput]), (measuresForAnExample[:error])]
    end
    return anArrayOfVectors
  end

  def test_clusterAllResponses1
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    @metricRecorder.verify
  end

  def printDebugInfo
    puts "#{@metricRecorder.vectorizeEpochMeasures}"
    @aFlockingNeuron.clusterer.clusters.each_with_index do |aCluster, clusterNumber|
      puts "Cluster #{clusterNumber} center= #{aCluster.center};  dispersion= #{aCluster.dispersion(@vectorizedTestData)}"
      puts "\t\t\tCluster Weightings= #{aCluster.exampleMembershipWeightsForCluster}"
    end
  end

  def test_clusterAllResponses2
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    distanceBetween = (Vector[-1.0, -1.0] - @clusters[0].center).r
    assert_in_delta(0.0, distanceBetween, Tolerance, "error in cluster 0's center determination")
  end

  def test_clusterAllResponses3
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    distanceBetween = (Vector[1.0, 1.0] - @clusters[1].center).r
    assert_in_delta(0.0, distanceBetween, Tolerance, "error in cluster 1's center determination")
  end

  def test_clusterAllResponses4
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    assert_in_delta(0.0, @clusters[0].dispersion(@vectorizedTestData), Tolerance, "error in cluster 0's center determination")
  end

  def test_clusterAllResponses5
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    assert_in_delta(0.0, @clusters[1].dispersion(@vectorizedTestData), Tolerance, "error in cluster 1's center determination")
  end
end

class Test2OfFlockingNeuronAndDynamicClusterer < MiniTest::Unit::TestCase
  def setup
    srand(0)
    @testData = [{:netInput => -1.1, :error => -0.9}, {:netInput => -0.9, :error => -1.1},
                 {:netInput => 1.1, :error => 0.9}, {:netInput => 0.9, :error => 1.1}]

    @vectorizedTestData = self.convertEachHashesToVectors(@testData)

    args = {:typeOfClusterer => DynamicClusterer,
            :numberOfClusters => 2,
            :m => 2.0,
            :numExamples => 4,
            :exampleVectorLength => 2,
            :delta => 0.001,
            :maxNumberOfClusteringIterations => 100
    }

    @aFlockingNeuron = FlockingNeuron.new(args)

    @metricRecorder = MiniTest::Mock.new
    @aFlockingNeuron.metricRecorder= @metricRecorder

    @clusters = @aFlockingNeuron.clusterer.clusters
  end

  def convertEachHashesToVectors(anArray)
    anArrayOfVectors = anArray.collect do |measuresForAnExample|
      Vector[(measuresForAnExample[:netInput]), (measuresForAnExample[:error])]
    end
    return anArrayOfVectors
  end

  def test_clusterAllResponses1
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    @metricRecorder.verify
  end

  def printDebugInfo
    puts "#{@metricRecorder.vectorizeEpochMeasures}"
    @aFlockingNeuron.clusterer.clusters.each_with_index do |aCluster, clusterNumber|
      puts "Cluster #{clusterNumber} center= #{aCluster.center};  dispersion= #{aCluster.dispersion(@vectorizedTestData)}"
      puts "\t\t\tCluster Weightings= #{aCluster.exampleMembershipWeightsForCluster}"
    end
  end

  def test_clusterAllResponses2
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    distanceBetween = (Vector[-0.9987404049178501, -0.998740406992638] - @clusters[0].center).r
    assert_in_delta(0.0, distanceBetween, Tolerance, "error in cluster 0's center determination")
  end

  def test_clusterAllResponses3
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    distanceBetween = (Vector[0.9987404045346213, 0.9987404073756669] - @clusters[1].center).r
    assert_in_delta(0.0, distanceBetween, Tolerance, "error in cluster 1's center determination")
  end

  def test_clusterAllResponses4
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    assert_in_delta(0.1416086364049178, @clusters[0].dispersion(@vectorizedTestData), Tolerance, "error in cluster 0's center determination")
  end

  def test_clusterAllResponses5
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    assert_in_delta(0.1416086364049178, @clusters[1].dispersion(@vectorizedTestData), Tolerance, "error in cluster 1's center determination")
  end

  def test_calcLocalFlockingError1
    exampleNumber = 0
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @metricRecorder.expect(:withinEpochMeasures, @testData, [])

    @aFlockingNeuron.clusterAllResponses

    @aFlockingNeuron.exampleNumber = exampleNumber
    actualNetInput = (@vectorizedTestData[exampleNumber])[0]

    clustersApproximationOfLocationOfExamplesNetInput = @aFlockingNeuron.clusterer.estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster(exampleNumber)[0]
    expectedFlockingError = clustersApproximationOfLocationOfExamplesNetInput - actualNetInput

    @aFlockingNeuron.calcLocalFlockingError {}
    assert_equal(expectedFlockingError, @aFlockingNeuron.localFlockingError, "flocking error for given example and neuron are incorrect")

    @metricRecorder.verify
  end

  def test_calcLocalFlockingError2
    [1, 2, 3].each do |exampleNumber|
      @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
      @metricRecorder.expect(:withinEpochMeasures, @testData, [])

      @aFlockingNeuron.clusterAllResponses

      @aFlockingNeuron.exampleNumber = exampleNumber
      actualNetInput = (@vectorizedTestData[exampleNumber])[0]
      clustersApproximationOfLocationOfExamplesNetInput = @aFlockingNeuron.clusterer.estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster(exampleNumber)[0]
      expectedFlockingError = clustersApproximationOfLocationOfExamplesNetInput - actualNetInput

      @aFlockingNeuron.calcLocalFlockingError {}

      assert_equal(expectedFlockingError, @aFlockingNeuron.localFlockingError, "flocking error for given example and neuron are incorrect")
      @metricRecorder.verify
    end
  end
end

############################################################      N
class TestOfFlockingOutputNeuronAndDynamicClusterer < MiniTest::Unit::TestCase
  def setup
    srand(0)
    @testData = [{:netInput => -1.0, :error => -1.0}, {:netInput => -1.0, :error => -1.0},
                 {:netInput => 1.0, :error => 1.0}, {:netInput => 1.0, :error => 1.0}]

    @vectorizedTestData = self.convertEachHashToAVector(@testData)

    args = {:typeOfClusterer => DynamicClusterer,
            :numberOfClusters => 2,
            :m => 2.0,
            :numExamples => 4,
            :exampleVectorLength => 2,
            :delta => 0.001,
            :maxNumberOfClusteringIterations => 100
    }


    @aFlockingNeuron = FlockingOutputNeuron.new(args)

    @metricRecorder = MiniTest::Mock.new
    @aFlockingNeuron.metricRecorder= @metricRecorder

    @clusters = @aFlockingNeuron.clusterer.clusters
  end

  def convertEachHashToAVector(anArray)
    anArrayOfVectors = anArray.collect do |measuresForAnExample|
      Vector[(measuresForAnExample[:netInput]), (measuresForAnExample[:error])]
    end
    return anArrayOfVectors
  end

  def test_clusterAllResponses1
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    @metricRecorder.verify
  end

  def printDebugInfo
    puts "#{@metricRecorder.vectorizeEpochMeasures}"
    @aFlockingNeuron.clusterer.clusters.each_with_index do |aCluster, clusterNumber|
      puts "Cluster #{clusterNumber} center= #{aCluster.center};  dispersion= #{aCluster.dispersion(@vectorizedTestData)}"
      puts "\t\t\tCluster Weightings= #{aCluster.exampleMembershipWeightsForCluster}"
    end
  end

  def test_clusterAllResponses2
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    distanceBetween = (Vector[-1.0, -1.0] - @clusters[0].center).r
    assert_in_delta(0.0, distanceBetween, Tolerance, "error in cluster 0's center determination")
  end

  def test_clusterAllResponses3
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    distanceBetween = (Vector[1.0, 1.0] - @clusters[1].center).r
    assert_in_delta(0.0, distanceBetween, Tolerance, "error in cluster 1's center determination")
  end

  def test_clusterAllResponses4
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    assert_in_delta(0.0, @clusters[0].dispersion(@vectorizedTestData), Tolerance, "error in cluster 0's center determination")
  end

  def test_clusterAllResponses5
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    assert_in_delta(0.0, @clusters[1].dispersion(@vectorizedTestData), Tolerance, "error in cluster 1's center determination")
  end
end

class Test2OfFlockingOutputNeuronAndDynamicClusterer < MiniTest::Unit::TestCase
  def setup
    srand(0)
    @testData = [{:netInput => -1.1, :error => -0.9}, {:netInput => -0.9, :error => -1.1},
                 {:netInput => 1.1, :error => 0.9}, {:netInput => 0.9, :error => 1.1}]

    @vectorizedTestData = self.convertEachHashesToVectors(@testData)

    args = {:typeOfClusterer => DynamicClusterer,
            :numberOfClusters => 2,
            :m => 2.0,
            :numExamples => 4,
            :exampleVectorLength => 2,
            :delta => 0.001,
            :maxNumberOfClusteringIterations => 100
    }

    @aFlockingNeuron = FlockingOutputNeuron.new(args)

    @metricRecorder = MiniTest::Mock.new
    @aFlockingNeuron.metricRecorder= @metricRecorder

    @clusters = @aFlockingNeuron.clusterer.clusters
  end

  def convertEachHashesToVectors(anArray)
    anArrayOfVectors = anArray.collect do |measuresForAnExample|
      Vector[(measuresForAnExample[:netInput]), (measuresForAnExample[:error])]
    end
    return anArrayOfVectors
  end

  def test_clusterAllResponses1
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    @metricRecorder.verify
  end

  def printDebugInfo
    puts "#{@metricRecorder.vectorizeEpochMeasures}"
    @aFlockingNeuron.clusterer.clusters.each_with_index do |aCluster, clusterNumber|
      puts "Cluster #{clusterNumber} center= #{aCluster.center};  dispersion= #{aCluster.dispersion(@vectorizedTestData)}"
      puts "\t\t\tCluster Weightings= #{aCluster.exampleMembershipWeightsForCluster}"
    end
  end

  def test_clusterAllResponses2
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    distanceBetween = (Vector[-0.9987404049178501, -0.998740406992638] - @clusters[0].center).r
    assert_in_delta(0.0, distanceBetween, Tolerance, "error in cluster 0's center determination")
  end

  def test_clusterAllResponses3
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    distanceBetween = (Vector[0.9987404045346213, 0.9987404073756669] - @clusters[1].center).r
    assert_in_delta(0.0, distanceBetween, Tolerance, "error in cluster 1's center determination")
  end

  def test_clusterAllResponses4
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    assert_in_delta(0.1416086364049178, @clusters[0].dispersion(@vectorizedTestData), Tolerance, "error in cluster 0's center determination")
  end

  def test_clusterAllResponses5
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @aFlockingNeuron.clusterAllResponses
    assert_in_delta(0.1416086364049178, @clusters[1].dispersion(@vectorizedTestData), Tolerance, "error in cluster 1's center determination")
  end

  def test_calcLocalFlockingError1
    exampleNumber = 0
    @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
    @metricRecorder.expect(:withinEpochMeasures, @testData, [])

    @aFlockingNeuron.clusterAllResponses

    @aFlockingNeuron.exampleNumber = exampleNumber
    actualNetInput = (@vectorizedTestData[exampleNumber])[0]

    clustersApproximationOfLocationOfExamplesNetInput = @aFlockingNeuron.clusterer.estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster(exampleNumber)[0]
    expectedFlockingError = clustersApproximationOfLocationOfExamplesNetInput - actualNetInput

    @aFlockingNeuron.calcLocalFlockingError {}
    assert_equal(expectedFlockingError, @aFlockingNeuron.localFlockingError, "flocking error for given example and neuron are incorrect")

    @metricRecorder.verify
  end

  def test_calcLocalFlockingError2
    [1, 2, 3].each do |exampleNumber|
      @metricRecorder.expect(:vectorizeEpochMeasures, @vectorizedTestData, [])
      @metricRecorder.expect(:withinEpochMeasures, @testData, [])

      @aFlockingNeuron.clusterAllResponses

      @aFlockingNeuron.exampleNumber = exampleNumber
      actualNetInput = (@vectorizedTestData[exampleNumber])[0]
      clustersApproximationOfLocationOfExamplesNetInput = @aFlockingNeuron.clusterer.estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster(exampleNumber)[0]
      expectedFlockingError = clustersApproximationOfLocationOfExamplesNetInput - actualNetInput

      @aFlockingNeuron.calcLocalFlockingError {}

      assert_equal(expectedFlockingError, @aFlockingNeuron.localFlockingError, "flocking error for given example and neuron are incorrect")
      @metricRecorder.verify
    end
  end
end

