# weightedClustering_test.rb

require 'test/unit'
#require 'minitest/reporters'
#Test::Reporters.use!

require_relative '../lib/core/WeightedClustering'

Tolerance = 0.00001

############################################################
################# Mods for Testing and Test  Fixtures ######
class DynamicClusterer
  attr_accessor :numberOfClusters, :m, :delta, :maxNumberOfClusteringIterations, :clusters
  public :examplesFractionalMembershipInEachCluster, :forEachExampleDetermineItsFractionalMembershipInEachCluster, :recenterClusters,
         :determineClusterAssociatedWithExample, :withinClusterDispersion,
         :withinClusterDispersion, :withinClusterDispersionOfInputs
end

class Cluster
  attr_accessor :center, :m, :numExamples, :vectorLength, :exampleWeightings
end

######################  TESTS  TESTS    #####################
class TestCluster < Test::Unit::TestCase
  # Called before every test method runs. Can be used
  # to set up fixture information.
  def setup
    srand(0.0)
    seedsForDataArray = [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0]
    @data = seedsForDataArray.collect do |aScalar|
      Vector[aScalar, aScalar]
    end
    @aCluster = Cluster.new(m=2.0, numExamples=6, examplesVectorLength=2)
    @aCluster.calcCenterInVectorSpace(@data)
  end

  def test_calcCenter
    @aCluster.membershipWeightForEachExample = Array.new(@data.length, 1.0)
    center = @aCluster.calcCenterInVectorSpace(@data)
    assert_equal(Vector[0.0, 0.0], center, "Error1: center was not calculated properly")
    3.times do |i|
      @aCluster.membershipWeightForEachExample[i]= 0.0
    end
    center = @aCluster.calcCenterInVectorSpace(@data)
    assert_equal(Vector[1.5, 1.5], center, "Error2: center was not calculated properly")
  end

  def test_recenter
    @aCluster.membershipWeightForEachExample = Array.new(@data.length, 1.0)
    @aCluster.calcCenterInVectorSpace(@data)
    3.times do |i|
      @aCluster.membershipWeightForEachExample[i]= 0.0
    end
    expected = (Vector[1.5, 1.5]).r
    delta = expected * Tolerance
    distanceMoved = @aCluster.recenter!(@data)
    assert_in_delta(expected, distanceMoved, delta, "Test of Clustering System: Cluster recenter! method calculated wrong 'distance moved.'")
  end

  def test_ClusterInitialization
    cluster1 = Cluster.new(m=2.0, 5, 2)
    cluster2 = Cluster.new(m=2.0, 5, 2)
    refute_same(cluster1.calcCenterInVectorSpace(@data), cluster2.calcCenterInVectorSpace(@data), "Error: the centers of the 2 clusters are the same!!")
  end
end

class TestClusterDispersion1 < Test::Unit::TestCase
  # Called before every test method runs. Can be used
  # to set up fixture information.
  def setup
    @data = []
    @data << Vector[-3.0, 0.0] << Vector[-2.0, 0.0] << Vector[-1.0, 0.0] << Vector[0.0, 0.0] << Vector[1.0, 0.0]
    @aCluster = Cluster.new(2.0, 5, 2)
    @aCluster.membershipWeightForEachExample = Array.new(@data.length, 1.0)
    @aCluster.calcCenterInVectorSpace(@data)
  end

  def test_Dispersion
    center= @aCluster.calcCenterInVectorSpace(@data)
    assert_equal(Vector[-1.0, 0.0], center, "Error1: center was not calculated properly")

    expected = Math.sqrt(2.0)
    delta = expected * Tolerance
    dispersion = @aCluster.dispersion(@data)
    assert_in_delta(expected, dispersion, delta, "Error: Dispersion calculation incorrect")


    expected = Math.sqrt(2.0)
    delta = expected * Tolerance
    dispersionOfInputs = @aCluster.dispersionOfInputs(@data)
    assert_in_delta(expected, dispersionOfInputs, delta, "Error: DispersionOfInputs calculation incorrect")
  end
end

class TestClusterDispersion2 < Test::Unit::TestCase
  def setup
    @data = []
    @data << Vector[-3.0, 0.0] << Vector[-2.0, 0.0] << Vector[-1.0, 0.0] << Vector[0.0, 0.0] << Vector[1.0, 0.0]
    @aCluster = Cluster.new(2.0, 5, 2)
    @aCluster.membershipWeightForEachExample = Array.new(@data.length, 1.0)
    @aCluster.membershipWeightForEachExample[3] = 0.0
    @aCluster.membershipWeightForEachExample[4] = 0.0
    @center = @aCluster.calcCenterInVectorSpace(@data)
  end

  def test_Dispersion
    assert_equal(Vector[-2.0, 0.0], @center, "Error1: center was not calculated properly")

    expected = Math.sqrt(2.0/3.0)
    delta = expected * Tolerance
    dispersion = @aCluster.dispersion(@data)
    assert_in_delta(expected, dispersion, delta, "Error: Dispersion calculation incorrect")

    expected = Math.sqrt(2.0/3.0)
    delta = expected * Tolerance
    dispersionOfInputs = @aCluster.dispersionOfInputs(@data)
    assert_in_delta(expected, dispersionOfInputs, delta, "Error: DispersionOfInputs calculation incorrect")
  end
end

class TestClusterDispersion3 < Test::Unit::TestCase
  def setup
    @data = []
    a = 1.0 / Math.sqrt(2.0)
    @a = a
    @data << Vector[-3.0 * a, -3.0 * a] << Vector[-2.0 * a, -2.0 * a] << Vector[-1.0 * a, -1.0 * a] << Vector[0.0, 0.0] << Vector[a, a]
    @aCluster = Cluster.new(2.0, 5, 2)
    @aCluster.membershipWeightForEachExample = Array.new(@data.length, 1.0)
    @aCluster.calcCenterInVectorSpace(@data)
  end

  def test_Dispersion
    center= @aCluster.calcCenterInVectorSpace(@data)
    expected = -1.0 * @a
    delta = expected * Tolerance
    xPartOfCenter = center.to_a[0]
    assert_in_delta(expected, xPartOfCenter, delta.abs, "Error0: CENTER calculation incorrect")

    expected = Math.sqrt(2.0)
    delta = expected * Tolerance
    dispersion = @aCluster.dispersion(@data)
    assert_in_delta(expected, dispersion, delta, "Error1: Dispersion calculation incorrect")

    expected = Math.sqrt((2*((2.0*@a)**2.0) + 2.0*(@a**2.0))/5.0)
    delta = expected * Tolerance
    dispersionOfInputs = @aCluster.dispersionOfInputs(@data)
    assert_in_delta(expected, dispersionOfInputs, delta, "Error2: DispersionOfInputs calculation incorrect")
  end
end


class TestDynamicClustering < Test::Unit::TestCase
  def setup
    srand(0.0)
    @points = []
    @scaleFactor = 1.0
    @points << (@scaleFactor * Vector[-3.0, 0.0]) << (@scaleFactor * Vector[-2.0, 0.0]) << (@scaleFactor * Vector[-1.0, 0.0]) << (@scaleFactor * Vector[0.0, 0.0]) << (@scaleFactor * Vector[1.0, 0.0])
    @numberOfClusters = 2; @m=2.0; @numExamples = 5; @vectorLength = 2; @delta=0.001; @maxNumberOfClusteringIterations= 100
    @aClusterer = DynamicClusterer.new(
        {:numberOfClusters => @numberOfClusters,
         :m => @m,
         :numExamples => @numExamples,
         :exampleVectorLength => @vectorLength,
         :delta => @delta,
         :maxNumberOfClusteringIterations => @maxNumberOfClusteringIterations})
    @aClusterer.initializationOfClusterCenters(@points)
    @cluster0 = @aClusterer.clusters[0]
    @cluster1 = @aClusterer.clusters[1]
  end

  def test_initialization
    cluster0ExpectedCenter = Vector[-1.0981518336640481, 0.0]
    cluster0Actual = @cluster0.center
    distanceBetweenActualAndExpected = (cluster0Actual - cluster0ExpectedCenter).r
    delta = 0.0001
    assert_in_delta(0.0, distanceBetweenActualAndExpected, delta, "Error: Cluster 0 center too far away from expected position")
    cluster1ExpectedCenter = Vector[-0.6122303913842454, 0.0]
    cluster1Actual = @cluster1.center
    distanceBetweenActualAndExpected = (cluster1Actual - cluster1ExpectedCenter).r
    assert_in_delta(0.0, distanceBetweenActualAndExpected, delta, "Error: Cluster 1 center too far away from expected position")
  end

  def test_calcExampleMemberships
    @cluster0.center = @scaleFactor * Vector[1.000001, 0.0]
    @cluster1.center = @scaleFactor * Vector[-3.000001, 0.0]
    @aClusterer.forEachExampleDetermineItsFractionalMembershipInEachCluster(@points)
    expected = [6.249996876748005e-14, 0.100000119999984, 0.5, 0.8999998800000161, 0.9999999999999376]
    assert_equal(expected, @cluster0.membershipWeightForEachExample, "Error1: example weightings for cluster 0 were not calculated correctly")
    expected = [0.9999999999999376, 0.899999880000016, 0.5, 0.10000011999998398, 6.249996873972448e-14]
    assert_equal(expected, @cluster1.membershipWeightForEachExample, "Error2: example weightings for cluster 1 were not calculated correctly")
  end

  def test_recenterClusters
    @cluster0.center = @scaleFactor * Vector[1.000001, 0.0]
    @cluster1.center = @scaleFactor * Vector[-3.000001, 0.0]
    @aClusterer.forEachExampleDetermineItsFractionalMembershipInEachCluster(@points)
    @aClusterer.recenterClusters(@points)
    cluster0ExpectedCenter = Vector[0.35265701435265806, 0.0]
    cluster0Actual = @cluster0.center
    distanceBetweenActualAndExpected = (cluster0Actual - cluster0ExpectedCenter).r
    delta = 0.0001
    assert_in_delta(0.0, distanceBetweenActualAndExpected, delta, "Error: Cluster 0 center too far away from expected position")
    cluster1ExpectedCenter = Vector[-2.352657014352598, 0.0]
    cluster1Actual = @cluster1.center
    distanceBetweenActualAndExpected = (cluster1Actual - cluster1ExpectedCenter).r
    assert_in_delta(0.0, distanceBetweenActualAndExpected, delta, "Error: Cluster 1 center too far away from expected position")
  end

  def test_recenterAndCalcExampleMemberships
    @cluster0.center = @scaleFactor * Vector[1.000001, 0.0]
    @cluster1.center = @scaleFactor * Vector[-3.000001, 0.0]
    @aClusterer.forEachExampleDetermineItsFractionalMembershipInEachCluster(@points)
    @aClusterer.recenterClusters(@points)
    cluster0ExpectedCenter = Vector[0.35265701435265806, 0.0]
    cluster0Actual = @cluster0.center
    distanceBetweenActualAndExpected = (cluster0Actual - cluster0ExpectedCenter).r
    delta = 0.001
    assert_in_delta(0.0, distanceBetweenActualAndExpected, delta, "Error: Cluster 0 center too far away from expected position")
    cluster1ExpectedCenter = Vector[-2.352657014352598, 0.0]
    cluster1Actual = @cluster1.center
    distanceBetweenActualAndExpected = (cluster1Actual - cluster1ExpectedCenter).r
    assert_in_delta(0.0, distanceBetweenActualAndExpected, delta, "Error: Cluster 1 center too far away from expected position")
    @aClusterer.forEachExampleDetermineItsFractionalMembershipInEachCluster(@points)
    expected = [0.035941326907624534, 0.021975439959311133, 0.4999999999999778, 0.9780245600406804, 0.9640586730923806]
    assert_equal(expected, @cluster0.membershipWeightForEachExample, "Error1: example weightings for cluster 0 were not calculated correctly")
    expected = [0.9640586730923754, 0.9780245600406889, 0.5000000000000222, 0.021975439959319554, 0.035941326907619336]
    assert_equal(expected, @cluster1.membershipWeightForEachExample, "Error2: example weightings for cluster 1 were not calculated correctly")
  end

  def test_clusterData
    @cluster0.center = @scaleFactor * Vector[-0.5, 0.0] #Vector[1.000001, 0.0]
    @cluster1.center = @scaleFactor * Vector[-1.5000001, 0.0] # Vector[-3.000001, 0.0]
    @aClusterer.m = 2.0
    @aClusterer.forEachExampleDetermineItsFractionalMembershipInEachCluster(@points)
    dummyVariable, numberOfIterations = @aClusterer.clusterData(@points)
    expected = 9
    actual = numberOfIterations
    assert_equal(expected, actual, "Error: Incorrect number of iterations and/or failed to converge")
  end

  def test_clusterData2
    @aClusterer.forEachExampleDetermineItsFractionalMembershipInEachCluster(@points)
    dummyVariable, numberOfIterations = @aClusterer.clusterData(@points)
    expected = 12
    actual = numberOfIterations
    assert_equal(expected, actual, "Error: Incorrect number of iterations and/or failed to converge")
  end

  def test_determineClusterAssociatedWithExample
    @aClusterer.forEachExampleDetermineItsFractionalMembershipInEachCluster(@points)
    dummyVariable, numberOfIterations = @aClusterer.clusterData(@points)
    ################ prior 2 lines are assumed to do the right thing! ##############################

    # @aClusterer.clusters.each { |cluster| puts "cluster=\t#{cluster}\n" }
    # puts "********************************************************"
    exampleNumber = 0
    aCluster = @aClusterer.determineClusterAssociatedWithExample(exampleNumber)
    # puts aCluster.to_s
    # puts
    actual = aCluster
    expected = @cluster0
    assert_equal(expected, actual, "Error: Wrong Cluster Chosen")
    actual = @cluster0.membershipWeightForEachExample[exampleNumber] > @cluster1.membershipWeightForEachExample[exampleNumber]
    expected = true
    assert_equal(expected, actual, "Error: Wrong Cluster Chosen")

    exampleNumber = 4
    aCluster = @aClusterer.determineClusterAssociatedWithExample(exampleNumber)
    actual = aCluster
    expected = @cluster1
    assert_equal(expected, actual, "Error: Wrong Cluster Chosen")
    actual = @cluster1.membershipWeightForEachExample[exampleNumber] > @cluster0.membershipWeightForEachExample[exampleNumber]
    expected = true
    assert_equal(expected, actual, "Error: Wrong Cluster Chosen")

    exampleNumber = 2
    aCluster = @aClusterer.determineClusterAssociatedWithExample(exampleNumber)
    actual = aCluster
    expected = @cluster0
    assert_equal(expected, actual, "Error: Wrong Cluster Chosen")
    actual = @cluster0.membershipWeightForEachExample[exampleNumber] > @cluster1.membershipWeightForEachExample[exampleNumber]
    expected = true
    assert_equal(expected, actual, "Error: Wrong Cluster Chosen")
  end
end


class TestDynamicClusteringB < Test::Unit::TestCase
  class DummyCluster
    attr_accessor :center
  end

  class DummyClusterer < DynamicClusterer
    def clusters
      cluster1 = DummyCluster.new
      cluster1.center = Vector[-1.0, 0.0]
      cluster2 = DummyCluster.new
      cluster2.center = Vector[1.0, 0.0]
      return [cluster1, cluster2]
    end

    def examplesFractionalMembershipInEachCluster(pointNumber)
      return [0.5, 0.5]
    end
  end

  def setup
    srand(0.0)
    @points = []
    @scaleFactor = 1.0
    @points << (@scaleFactor * Vector[-3.0, 0.0]) << (@scaleFactor * Vector[-2.0, 0.0]) << (@scaleFactor * Vector[-1.0, 0.0]) << (@scaleFactor * Vector[0.0, 0.0]) << (@scaleFactor * Vector[1.0, 0.0])
    @numberOfClusters = 2; @m=2.0; @numExamples = 5; @vectorLength = 2; @delta=0.001; @maxNumberOfClusteringIterations= 100
    @aClusterer = DummyClusterer.new(
        {:numberOfClusters => @numberOfClusters,
         :m => @m,
         :numExamples => @numExamples,
         :exampleVectorLength => @vectorLength,
         :delta => @delta,
         :maxNumberOfClusteringIterations => @maxNumberOfClusteringIterations}
    )
    @aClusterer.initializationOfClusterCenters(@points)
    @cluster0 = @aClusterer.clusters[0]
    @cluster1 = @aClusterer.clusters[1]
  end

  def test_estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster
    actual = @aClusterer.estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster(1)
    assert_equal(Vector[0.0, 0.0], actual, "Error in estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster")
  end
end


class TestDynamicClusteringC < Test::Unit::TestCase
  class DummyCluster
    attr_accessor :center
  end

  class DummyClusterer < DynamicClusterer
    def clusters
      cluster1 = DummyCluster.new
      cluster1.center = Vector[-0.5, 0.0]
      cluster2 = DummyCluster.new
      cluster2.center = Vector[1.0, 0.0]
      return [cluster1, cluster2]
    end

    def examplesFractionalMembershipInEachCluster(pointNumber)
      return [0.5, 0.5]
    end
  end

  def setup
    srand(0.0)
    @points = []
    @scaleFactor = 1.0
    @points << (@scaleFactor * Vector[-3.0, 0.0]) << (@scaleFactor * Vector[-2.0, 0.0]) << (@scaleFactor * Vector[-1.0, 0.0]) << (@scaleFactor * Vector[0.0, 0.0]) << (@scaleFactor * Vector[1.0, 0.0])
    @numberOfClusters = 2; @m=2.0; @numExamples = 5; @vectorLength = 2; @delta=0.001; @maxNumberOfClusteringIterations= 100
    @aClusterer = DummyClusterer.new(
        {:numberOfClusters => @numberOfClusters,
         :m => @m,
         :numExamples => @numExamples,
         :exampleVectorLength => @vectorLength,
         :delta => @delta,
         :maxNumberOfClusteringIterations => @maxNumberOfClusteringIterations}
    )
    @aClusterer.initializationOfClusterCenters(@points)
    @cluster0 = @aClusterer.clusters[0]
    @cluster1 = @aClusterer.clusters[1]
  end

  def test_estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster
    actual = @aClusterer.estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster(1)
    assert_equal(Vector[0.25, 0.0], actual, "Error in estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster")
  end
end

class TestDynamicClusteringD < Test::Unit::TestCase
  class DummyCluster
    attr_accessor :center
  end

  class DummyClusterer < DynamicClusterer
    def clusters
      cluster1 = DummyCluster.new
      cluster1.center = Vector[-0.5, 0.0]
      cluster2 = DummyCluster.new
      cluster2.center = Vector[1.0, 0.0]
      return [cluster1, cluster2]
    end

    def examplesFractionalMembershipInEachCluster(pointNumber)
      return [0.666666666666666666666666666666666, 0.3333333333333333333333333333333333]
    end
  end

  def setup
    srand(0.0)
    @points = []
    @scaleFactor = 1.0
    @points << (@scaleFactor * Vector[-3.0, 0.0]) << (@scaleFactor * Vector[-2.0, 0.0]) << (@scaleFactor * Vector[-1.0, 0.0]) << (@scaleFactor * Vector[0.0, 0.0]) << (@scaleFactor * Vector[1.0, 0.0])
    @numberOfClusters = 2; @m=2.0; @numExamples = 5; @vectorLength = 2; @delta=0.001; @maxNumberOfClusteringIterations= 100
    @aClusterer = DummyClusterer.new(
        {:numberOfClusters => @numberOfClusters,
         :m => @m,
         :numExamples => @numExamples,
         :exampleVectorLength => @vectorLength,
         :delta => @delta,
         :maxNumberOfClusteringIterations => @maxNumberOfClusteringIterations}
    )
    @aClusterer.initializationOfClusterCenters(@points)
    @cluster0 = @aClusterer.clusters[0]
    @cluster1 = @aClusterer.clusters[1]
  end

  def test_estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster
    actual = @aClusterer.estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster(1)
    assert_equal(Vector[0.0, 0.0], actual, "Error in estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster")
  end
end


class TestDynamicClusteringRecent < Test::Unit::TestCase
  class DummyCluster
    attr_accessor :center
  end

  class DummyClusterer < DynamicClusterer
    def clusters
      cluster1 = DummyCluster.new
      cluster1.center = Vector[1.0, 0.0]
      cluster2 = DummyCluster.new
      cluster2.center = Vector[-1.0, 0.0]
      return [cluster1, cluster2]
    end

    def examplesFractionalMembershipInEachCluster(pointNumber)
      return [0.9, 0.0]
    end
  end

  def setup
    srand(0.0)
    @points = []
    @scaleFactor = 1.0
    @points << (@scaleFactor * Vector[3.0, 0.0]) << (@scaleFactor * Vector[2.0, 0.0]) << (@scaleFactor * Vector[1.0, 0.0]) << (@scaleFactor * Vector[0.0, 0.0]) << (@scaleFactor * Vector[-1.0, 0.0])
    @numberOfClusters = 2; @m=2.0; @numExamples = 5; @vectorLength = 2; @delta=0.001; @maxNumberOfClusteringIterations= 100
    @aClusterer = DummyClusterer.new(
        {:numberOfClusters => @numberOfClusters,
         :m => @m,
         :numExamples => @numExamples,
         :exampleVectorLength => @vectorLength,
         :delta => @delta,
         :maxNumberOfClusteringIterations => @maxNumberOfClusteringIterations}
    )
    @aClusterer.initializationOfClusterCenters(@points)
    @cluster0 = @aClusterer.clusters[0]
    @cluster1 = @aClusterer.clusters[1]
  end

  #def test_estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster
  #  actual = @aClusterer.estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster(1)
  #  assert_equal(Vector[0.0, 0.0], actual, "Error in estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster")
  #end

  def test_pointsTargetForIterationInFuzzyClustering
    actual = @aClusterer.pointsTargetForIterationInFuzzyClustering(2, @points)
    assert_equal(Vector[0.0, 0.0], actual, "Error in estimatePointsClusterCenterFromItsFractionalMembershipToEachCluster")
  end

end





