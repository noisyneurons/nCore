## NNMeasure.rb

require_relative '../core/NeuralParts'
require_relative 'CorePlottingCode'

################################### TrainingAndMeasurementTriggers ##########################
# This effectively the "VIEW" base class for the MVC architecture.
class TrainingAndMeasurementTriggers
  attr_reader :network, :trainer, :theNeuronToExamine, :specification

  def initialize(trainer, numberOfEpochsBetweenMeasures, indexToLayer, indexToNeuron, specification=nil)
    @trainer = trainer
    @exampleNumberForMeasurements = trainer.exampleNumber unless (trainer.exampleNumberForMeasurements.nil?)
    @network = trainer.network
    @indexToLayer = indexToLayer
    @indexToNeuron = indexToNeuron
    @numberOfEpochsBetweenMeasures = numberOfEpochsBetweenMeasures
    @specification = specification
    @measures = []
    theLayerToExamine = @network.nnComponents[@indexToLayer] unless (@indexToLayer.nil?)
    @theNeuronToExamine = theLayerToExamine.nnComponents[@indexToNeuron] unless (@indexToNeuron.nil?)
    @deviceSetup = "gif font arial 18 size 1024,768 xffffff x000000 x404040 xff0000 xffa500 x66cdaa xcdb5cd xadd8e6 x0000ff xdda0dd x9500d3 animate delay 5"
    @plotName = "#{@specification.experimentName}#{self.class.to_s}Layer#{@indexToLayer}Neuron#{@indexToNeuron}"
  end

  # Interface stubs
  def beginningOfTrainingMeasures
  end

  def beginningOfEpochProcessingMeasures
    @recordThisEpoch = (@trainer.epochNumber.modulo(@numberOfEpochsBetweenMeasures) == 0) # TODO should this statement be put in Trainer class.
  end

  def beginningOfFlockingMeasures
  end

  def flockingMeasures
  end

  def endOfExampleProcessingMeasures
  end

  def endOfFlockingMeasures
  end

  def endOfEpochProcessingMeasures
  end

  def endOfTrainingMeasures
  end
end

################################### Tester ###################################################
class Tester < TrainingAndMeasurementTriggers
  # attr_reader :numOutputs, :evaluationLayer, :network

  def initialize(testExampleList, trainerOfNetwork, numberOfEpochsBetweenMeasures, specification=nil)
    super(trainerOfNetwork, numberOfEpochsBetweenMeasures, nil, nil, specification)
    @measures = []
    @testExampleList = testExampleList
    @evaluationLayer = @network.evaluationLayerForMetrics
    @numOutputs = @evaluationLayer.numOutputs
    @deviceSetup = "gif font arial 18 size 1024,768 xffffff x000000 x404040 xff0000 xffa500 x66cdaa xcdb5cd xadd8e6 x0000ff xdda0dd x9500d3"
    @plotName = "#{@specification.experimentName}#{self.class.to_s}"
  end

  def test
    squaredError = 0.0
    @testExampleList.each do |example|
      @network.driveNetworkWithExample(example)
      @network.propagate
      squaredError += @evaluationLayer.networksErrorOnExample
    end
    testingMSE = squaredError / (@testExampleList.length * @numOutputs)
    return testingMSE
  end

  def endOfEpochProcessingMeasures
    @measures << [@trainer.epochNumber, @trainer.trainingMSE, self.test] if (@recordThisEpoch)
  end

  def endOfTrainingMeasures
    epochNumber = []
    trainingMSE = []
    testingMSE = []

    @measures.each do |withinEpochMeasures|
      epochNumber << withinEpochMeasures[0]
      trainingMSE << withinEpochMeasures[1]
      testingMSE << withinEpochMeasures[2]
    end

    aPlotter = Plotter.new(title="Training and Test Error for Experiment: #{@specification.experimentName}", xLabel="Epoch Number", yLabel="Training and Test Error",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}", @deviceSetup)
    aPlotter.plotTrainingAndTestError(epochNumber, trainingMSE, testingMSE)
  end
end

################################### HyperplaneAngle ##########################################
class HyperplaneAngle < TrainingAndMeasurementTriggers
  def initialize(trainer, numberOfEpochsBetweenMeasures, indexToLayer, indexToNeuron, specification=nil)
    super(trainer, numberOfEpochsBetweenMeasures, indexToLayer, indexToNeuron, specification)
    @theFirstLinkToExamine = @theNeuronToExamine.nnComponents[0]
    @theSecondLinkToExamine = @theNeuronToExamine.nnComponents[1]
  end

  def endOfEpochProcessingMeasures
    firstWeight = @theFirstLinkToExamine.weight
    secondWeight = @theSecondLinkToExamine.weight
    angleInRadians = Math.atan(firstWeight/secondWeight)
    averageOfAbsoluteWeights = (firstWeight.abs + secondWeight.abs) / 2.0
    @measures << [@trainer.epochNumber, angleInRadians, averageOfAbsoluteWeights] if (@recordThisEpoch)
  end

  def endOfTrainingMeasures
    epochNumber = []
    angleInRadians = []
    averageOfAbsoluteWeights = []

    @measures.each do |withinEpochMeasures|
      epochNumber << withinEpochMeasures[0]
      angleInRadians << withinEpochMeasures[1]
      averageOfAbsoluteWeights << withinEpochMeasures[2]
    end

    aPlotter = Plotter.new(title="Hyperplane Angle/Sharpness; Layer #{@indexToLayer}; Neuron #{@indexToNeuron}; Exp: #{@specification.experimentName}", xLabel="Epoch Number", yLabel="Hyperplane Angle (radians) & Ave Weight",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}", @deviceSetup)
    aPlotter.plotHyperplaneMeasuresVsEpoch(epochNumber, angleInRadians, averageOfAbsoluteWeights)
  end
end

################################### BPErrorVsNetInput #########################################
class BPErrorVsNetInput < TrainingAndMeasurementTriggers
  def beginningOfEpochProcessingMeasures
    super
    @measuresWithinAnEpoch = [] if (@recordThisEpoch)
  end

  def endOfExampleProcessingMeasures
    if (@recordThisEpoch)
      if (@trainer.flockIterationNumber == (@specification.numberOfFlockingIterations-1))
        @measuresWithinAnEpoch << [@theNeuronToExamine.netInput, @theNeuronToExamine.error]
        # std("measures within an epoch", @measuresWithinAnEpoch)
      end
    end
  end

  def endOfEpochProcessingMeasures
    @measures << @measuresWithinAnEpoch if (@recordThisEpoch)
  end

  def endOfTrainingMeasures
    xMeasurementsForEachEpoch = []
    yMeasurementsForEachEpoch = []

    @measures.each do |withinEpochMeasures|
      xMeasurementsForEachExampleInOneEpoch = []
      yMeasurementsForEachExampleInOneEpoch = []
      withinEpochMeasures.each do |measurementsForOneExample|
        xMeasurementsForEachExampleInOneEpoch << measurementsForOneExample[0]
        yMeasurementsForEachExampleInOneEpoch << measurementsForOneExample[1]
      end
      xMeasurementsForEachEpoch << xMeasurementsForEachExampleInOneEpoch
      yMeasurementsForEachEpoch << yMeasurementsForEachExampleInOneEpoch
    end

    aPlotter = Plotter.new(title="LayerInx: #{@indexToLayer}; NeuronIdx: #{@indexToNeuron}; Experiment: #{@specification.experimentName}", xLabel="Net Input", yLabel="BackProp & Flocking Error",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}", @deviceSetup)
    aPlotter.plotAnAnimationWithFixedScales(xMeasurementsForEachEpoch, yMeasurementsForEachEpoch)
  end
end

################################### VectorBPErrorVsNetInput ###################################
class VectorBPErrorVsNetInput < BPErrorVsNetInput
  def beginningOfTrainingMeasures
    @dataStore = DataStoreAndDeltaStore.new
    @dataStore.beginningOfTrainingMeasures
  end

  def endOfExampleProcessingMeasures
    if (@recordThisEpoch)
      if (@trainer.flockIterationNumber == (@specification.numberOfFlockingIterations-1))
        @dataStore.addMeasure(@theNeuronToExamine.netInput)
        @dataStore.addMeasure(@theNeuronToExamine.error)
        @dataStore.endOfExampleProcessingMeasures
      end
    end
  end

  def endOfEpochProcessingMeasures
    @dataStore.endOfEpochProcessingMeasures if (@recordThisEpoch)
  end

  def endOfTrainingMeasures
    arrayOfEpochsWithEachMeasureInAnArray = @dataStore.endOfTrainingMeasures # four arrays: x deltax y deltay

    aPlotter = Plotter.new(title="LayerInx: #{@indexToLayer}; NeuronIdx: #{@indexToNeuron}; Experiment: #{@specification.experimentName}", xLabel="Net Input", yLabel="Vector BackProp Error",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}", @deviceSetup)

    xMeasurementsForEachEpoch = arrayOfEpochsWithEachMeasureInAnArray[0]
    deltaxMeasurementsForEachEpoch = arrayOfEpochsWithEachMeasureInAnArray[1]
    yMeasurementsForEachEpoch = arrayOfEpochsWithEachMeasureInAnArray[2]
    deltayMeasurementsForEachEpoch = arrayOfEpochsWithEachMeasureInAnArray[3]

    aPlotter.plotAnAnimationWithFixedScales(xMeasurementsForEachEpoch, yMeasurementsForEachEpoch, deltaxMeasurementsForEachEpoch, deltayMeasurementsForEachEpoch)
  end
end

################################### ContourPlotOfIOFunction ###################################
class ContourPlotOfIOFunction < TrainingAndMeasurementTriggers
  def beginningOfTrainingMeasures
    @inputValues = Array.new.fill(0, 41) { |i| ((i * 0.1) - 2.0) }
  end

  def recordIOMeasurements
    x0 = []; x1=[]; outputForIOFunction=[]
    @inputValues.each do |x0Val|
      @inputValues.each do |x1Val|
        x0 << x0Val
        x1 << x1Val
        outputForIOFunction << inputOutputFunction(x0Val, x1Val)
      end
    end
    return [x0, x1, outputForIOFunction]
  end

  def inputOutputFunction(x0, x1)
    @network.inputLayers[0].nnComponents[0].output = x0
    @network.inputLayers[0].nnComponents[1].output = x1
    @network.propagate
    return @theNeuronToExamine.output
  end

  def endOfEpochProcessingMeasures
    @measures << recordIOMeasurements if (@trainer.epochNumber.modulo(@numberOfEpochsBetweenMeasures) == 0)
  end

  def endOfTrainingMeasures
    x=[]; y=[]; z=[]
    @measures.each do |ioMeasureAtEndOfAnEpoch|
      x << ioMeasureAtEndOfAnEpoch[0]
      y << ioMeasureAtEndOfAnEpoch[1]
      z << ioMeasureAtEndOfAnEpoch[2]
    end

    @plotName = "#{@specification.experimentName}#{self.class.to_s}"

    aPlotter = Plotter.new("Input-Output Contour Plot for Experiment: #{@specification.experimentName}", xLabel="X input value", yLabel="Y input value",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}", @deviceSetup)
    aPlotter.plotAContourAnimation(x, y, z)
  end
end

##################  The following 3 classes were moved here from DemoPreTree  ############
######################## These 3 are related to Flocking #################################  # TODO Perhaps these 3 should be moved to Flocking Measures file??
################################### BPandFlockingError ###################################
class BPandFlockingError < TrainingAndMeasurementTriggers
  def beginningOfEpochProcessingMeasures
    super
    @measuresWithinAnEpoch = [] if (@recordThisEpoch)
  end

  def endOfExampleProcessingMeasures
    if (@recordThisEpoch)
      flockIterationNumber = @trainer.flockIterationNumber
      if ((flockIterationNumber == 0) || (flockIterationNumber == @specification.numberOfFlockingIterations-1))
        exampleNumber = network.example.exampleNumber
        numNeurons = network.layersForLocalFlocking[0].nnComponents.length
        std("numNeuronsInLayer=\t", numNeurons) if (exampleNumber == 0)
        cluster = @theNeuronToExamine.clusterer.determineClusterAssociatedWithExample(exampleNumber)
        clusterNumber = cluster.clusterNumber
        #       @measuresWithinAnEpoch << [@theNeuronToExamine.examplesFlockingError.abs, @theNeuronToExamine.error.abs]
        @measuresWithinAnEpoch << [@theNeuronToExamine.examplesFlockingError, @theNeuronToExamine.error, clusterNumber]
      end
    end
  end

  def endOfEpochProcessingMeasures
    @measures << @measuresWithinAnEpoch if (@recordThisEpoch)
  end

  def endOfTrainingMeasures
    xMeasurementsForEachEpoch = []
    yMeasurementsForEachEpoch = []
    clusterNumberForEachEpoch = []

    @measures.each do |withinEpochMeasures|
      xMeasurementsForEachExampleInOneEpoch = []
      yMeasurementsForEachExampleInOneEpoch = []
      clusterNumberForEachExampleInOneEpoch = []
      withinEpochMeasures.each do |measurementsForOneExample|
        xMeasurementsForEachExampleInOneEpoch << measurementsForOneExample[0]
        yMeasurementsForEachExampleInOneEpoch << measurementsForOneExample[1]
        clusterNumberForEachExampleInOneEpoch << measurementsForOneExample[2]
      end
      xMeasurementsForEachEpoch << xMeasurementsForEachExampleInOneEpoch
      yMeasurementsForEachEpoch << yMeasurementsForEachExampleInOneEpoch
      clusterNumberForEachEpoch << clusterNumberForEachExampleInOneEpoch
    end

    aPlotter = Plotter.new(title="BP & Flock Error; Layer #{@indexToLayer}; Neuron #{@indexToNeuron}; Exp: #{@specification.experimentName}", xLabel="Absolute Flocking Error", yLabel="BackProp & Flocking Error",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}", @deviceSetup)
    aPlotter.plotAColorCodedAnimationWithFixedScales(xMeasurementsForEachEpoch, yMeasurementsForEachEpoch, clusterNumberForEachEpoch)
  end
end

###################### Changes in Cluster NetInputs due to Flocking and ErrorBP  ###################################
class NetInputsDueToFlockingAndBP < TrainingAndMeasurementTriggers
  def beginningOfEpochProcessingMeasures
    super
    @recordThisEpoch = false if (@trainer.epochNumber==0)
    if (@recordThisEpoch)
      @withinEpochHash = Hash.new
      @withinEpochHash[:epochNumber] = @trainer.epochNumber
      @clusters = @theNeuronToExamine.clusterer.clusters
      @withinEpochHash[:startOfEpoch] = (@clusters.collect { |cluster| cluster.netInput }).sort
    end
  end

  def beginningOfFlockingMeasures
    @withinEpochHash[:afterBP] = (@clusters.collect { |cluster| cluster.netInput }).sort if (@recordThisEpoch)
  end

  def endOfFlockingMeasures
    @withinEpochHash[:afterBPAndFlocking] = (@clusters.collect { |cluster| cluster.netInput }).sort if (@recordThisEpoch)
  end

  def endOfEpochProcessingMeasures
    if (@recordThisEpoch)
      @measures << @withinEpochHash
    end
  end

  def endOfTrainingMeasures
    epochNumber = []
    initialNetInput = []
    netInputAfterBP = []
    netInputAfterBPAndFlocking = []

    clusterNumber = 0
    @measures.each do |epochHash|
      epochNumber << epochHash[:epochNumber]
      initialNetInput << epochHash[:startOfEpoch][clusterNumber]
      netInputAfterBP << epochHash[:afterBP][clusterNumber]
      netInputAfterBPAndFlocking << epochHash[:afterBPAndFlocking][clusterNumber]
    end

    aPlotter = Plotter.new(title="NetInput vs Epoch ; Layer #{@indexToLayer}; Neuron #{@indexToNeuron}; Exp: #{@specification.experimentName}", xLabel="Epoch Number", yLabel="Net Input",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}", @deviceSetup)
    aPlotter.plotNetInputVsEpoch(epochNumber, initialNetInput, netInputAfterBP, netInputAfterBPAndFlocking)

    epochNumber = []
    changeFromInitialToAfterBP = []
    changeFromInitialToAfterBPAndFlocking = []

    @measures.each do |epochHash|
      epochNumber << epochHash[:epochNumber]
      changeFromInitialToAfterBP << (epochHash[:afterBP][clusterNumber] - epochHash[:startOfEpoch][clusterNumber])
      changeFromInitialToAfterBPAndFlocking << (epochHash[:afterBPAndFlocking][clusterNumber] - epochHash[:startOfEpoch][clusterNumber])
    end

    aPlotter = Plotter.new(title="Changes in NetInput vs Epoch ; Layer #{@indexToLayer}; Neuron #{@indexToNeuron}; Exp: #{@specification.experimentName}", xLabel="Epoch Number", yLabel="Change in Net Input",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}ChngLin", @deviceSetup)
    aPlotter.plotChangeInNetInputVsEpoch(epochNumber, changeFromInitialToAfterBP, changeFromInitialToAfterBPAndFlocking, logY=false)


    aPlotter = Plotter.new(title="Changes in NetInput vs Epoch ; Layer #{@indexToLayer}; Neuron #{@indexToNeuron}; Exp: #{@specification.experimentName}", xLabel="Epoch Number", yLabel="Change in Net Input",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}ChngLog", @deviceSetup)
    changeFromInitialToAfterBP.collect! { |e| e.abs }
    changeFromInitialToAfterBPAndFlocking.collect! { |e| e.abs }
    aPlotter.plotChangeInNetInputVsEpoch(epochNumber, changeFromInitialToAfterBP, changeFromInitialToAfterBPAndFlocking, logY=true)

  end
end

################################### DispersionBeforeAfterFlocking ###################################

class DispersionBeforeAfterFlocking < TrainingAndMeasurementTriggers
  def beginningOfEpochProcessingMeasures
    super
    @measuresWithinAnEpoch = [] if (@recordThisEpoch)
  end

  def beginningOfFlockingMeasures
    if (@recordThisEpoch)
      dispersionOfInputs = calcDispersion()
      @measuresWithinAnEpoch << @trainer.epochNumber << dispersionOfInputs
    end
  end

  def endOfFlockingMeasures
    if (@recordThisEpoch)
      dispersionOfInputs = calcDispersion()
      @measuresWithinAnEpoch << dispersionOfInputs
    end
  end

  def calcDispersion
    points = @theNeuronToExamine.withinEpochMeasures
    dispersionOfInputs = @theNeuronToExamine.clusterer.withinClusterDispersionOfInputs(points)
    return dispersionOfInputs
  end

  def endOfEpochProcessingMeasures
    @measures << @measuresWithinAnEpoch if (@recordThisEpoch)
  end

  def endOfTrainingMeasures
    epochNumber = []
    beginningDispersion = []
    endingDispersion = []

    @measures.each do |withinEpochMeasures|
      epochNumber << withinEpochMeasures[0]
      beginningDispersion << withinEpochMeasures[1]
      endingDispersion << withinEpochMeasures[2]
    end

    aPlotter = Plotter.new(title="Input Dispersion Before/After ; Layer #{@indexToLayer}; Neuron #{@indexToNeuron}; Exp: #{@specification.experimentName}", xLabel="Epoch Number", yLabel="Flock Dispersion (of Neurons Inputs)",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}", @deviceSetup)
    aPlotter.plotDispersionVsEpoch(epochNumber, beginningDispersion, endingDispersion)
  end
end

################################### WeightMagnitudeBeforeAfterFlocking ###################################

class WeightMagnitudeBeforeAfterFlocking < TrainingAndMeasurementTriggers
  def beginningOfEpochProcessingMeasures
    super
    @measuresWithinAnEpoch = [] if (@recordThisEpoch)
  end

  def beginningOfFlockingMeasures
    @measuresWithinAnEpoch << @trainer.epochNumber << @theNeuronToExamine.magnitudeOfAllInputWeights if (@recordThisEpoch)
  end

  def endOfFlockingMeasures
    @measuresWithinAnEpoch << @theNeuronToExamine.magnitudeOfAllInputWeights if (@recordThisEpoch)
  end

  def endOfEpochProcessingMeasures
    @measures << @measuresWithinAnEpoch if (@recordThisEpoch)
  end

  def endOfTrainingMeasures
    epochNumber = []
    beginningWeightMagnitude = []
    endingWeightMagnitude = []

    @measures.each do |withinEpochMeasures|
      epochNumber << withinEpochMeasures[0]
      beginningWeightMagnitude << withinEpochMeasures[1]
      endingWeightMagnitude << withinEpochMeasures[2]
    end

    aPlotter = Plotter.new(title="Weight Mag Bef/Aft Flocking; Layer #{@indexToLayer}; Neuron #{@indexToNeuron}; Exp: #{@specification.experimentName}", xLabel="Epoch Number", yLabel="Magnitude of Weights",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}", @deviceSetup)
    aPlotter.plotWeightMagnitudeVsEpoch(epochNumber, beginningWeightMagnitude, endingWeightMagnitude)
  end
end
##################  The PRIOR 3 classes were moved here from DemoPreTree  ############

###################### Cluster Assigned Measurements of Flock and BP Error  ###################################
class BothErrorsVsNetInputAndFlockingViewClusterPlots < TrainingAndMeasurementTriggers

  def endOfExampleProcessingMeasures
    if (@recordThisEpoch)
      exampleNumber = network.example.exampleNumber
      cluster = @theNeuronToExamine.clusterer.determineClusterAssociatedWithExample(exampleNumber)

      unless (cluster.measures.has_key?('BeforeFlocking')) # Do we need to init hash values to arrays
                                                           # YES, Init hash values to arrays:
        cluster.measures['BeforeFlocking'] = []
        cluster.measures['EndOfFlocking'] = []
        cluster.measures['CenterOfFlock'] = []
      end
      if (@trainer.flockIterationNumber==0)
        cluster.measures['BeforeFlocking'] << [@theNeuronToExamine.netInput, @theNeuronToExamine.error, @theNeuronToExamine.examplesFlockingError]
        cluster.measures['CenterOfFlock'] << cluster.center
      end
      if (@trainer.flockIterationNumber == (@specification.numberOfFlockingIterations-1))
        cluster.measures['EndOfFlocking'] << [@theNeuronToExamine.netInput, @theNeuronToExamine.error, @theNeuronToExamine.examplesFlockingError]
      end
    end
  end

  def endOfEpochProcessingMeasures
    if (@recordThisEpoch)
      clusters = @theNeuronToExamine.clusters
      deepCopyOfClusters = Marshal.load(Marshal.dump(clusters))
      clearMeasures(clusters)
      @measures << deepCopyOfClusters
    end
  end

  def endOfTrainingMeasures
    aPlotter = Plotter.new(title="LayerInx: #{@indexToLayer}; NeuronIdx: #{@indexToNeuron}; Experiment: #{@specification.experimentName}", xLabel="Net Input", yLabel="BackProp & Flocking Error",
                           plotOutputFilenameBase = "./plots/#{specification.plotSubdirectory}/#{@plotName}", @deviceSetup)

    aPlotter.plot3AnAnimatedClusterPlotWithFixedScales(@measures)
  end

  private
  def clearMeasures(clusters)
    clusters.each do |aCluster|
      aCluster.measures['BeforeFlocking'].clear
      aCluster.measures['EndOfFlocking'].clear
      aCluster.measures['CenterOfFlock'].clear
    end
  end
end

#*********************************** UTILITY FUNCTIONS for NNMeasure ****************************************************

################################### DataStore And Re-Organization of that Data  ###################################

class DataStoreAndDeltaStore
  attr_accessor :multiEpochStorageArray, :epochStorageArray, :exampleMeasuresArray, :previousMeasures, :measureCounter, :firstEpoch

  def beginningOfTrainingMeasures
    @multiEpochStorageArray = []
    @epochStorageArray = []
    @exampleMeasuresArray = []
    @previousMeasures = []
    @measureCounter = 0
    @firstEpoch = true
  end

  def addMeasure(aMeasure)
    @exampleMeasuresArray << aMeasure
    unless (@firstEpoch)
      @exampleMeasuresArray << (aMeasure - @previousMeasures[@measureCounter]) # store delta
    else
      @exampleMeasuresArray << 0.0 # dummy 0.0 for delta because we are at the very beginning
    end
    @previousMeasures[@measureCounter] = aMeasure
    @measureCounter += 1
  end

  def endOfExampleProcessingMeasures
    @epochStorageArray << @exampleMeasuresArray
    @exampleMeasuresArray = []
  end

  def endOfEpochProcessingMeasures
    @firstEpoch = false
    @multiEpochStorageArray << @epochStorageArray
    @epochStorageArray = []
    @measureCounter = 0
  end

  def endOfTrainingMeasures
    #STDOUT.puts "For Vector, @multiEpochStorageArray.length #{@multiEpochStorageArray.length}\n\n\n"; STDOUT.flush
    eachMeasureInAnArray = []
    arrayContainingASingleMeasureOrganizedByEpochSubArrays = []
    numberOfMeasuresPerExample = @multiEpochStorageArray[0][0].length
    #STDOUT.puts "For Vector, numberOfMeasuresPerExample #{numberOfMeasuresPerExample}\n\n\n"; STDOUT.flush

    numberOfMeasuresPerExample.times { |i| eachMeasureInAnArray << [] }

    # The following is too complicated... it used to reorganize the data into gnuplots specs for animated vector plots
    @multiEpochStorageArray.each do |epochArray|
      withinEpochMeasurementArray = []
      numberOfMeasuresPerExample.times { |i| withinEpochMeasurementArray << [] }
      epochArray.each do |anExample|
        anExample.each_with_index do |aMeasure, indexToMeasureArray|
          withinEpochMeasurementArray[indexToMeasureArray] << aMeasure
        end
      end
      eachMeasureInAnArray.each_with_index do |arrayContainingASingleMeasureOrganizedByEpochSubArrays, measureNumber|
        arrayContainingASingleMeasureOrganizedByEpochSubArrays << withinEpochMeasurementArray[measureNumber]
      end
    end
    return eachMeasureInAnArray # four arrays: x deltax y deltay
  end

  def to_s
    description = "DataStore Class"
    description += "@multiEpochStorageArray= #{@multiEpochStorageArray}\n"
    description += "@epochStorageArray= #{@epochStorageArray}\n"
    description += "@exampleMeasuresArray= #{@exampleMeasuresArray}\n"
    description += "@previousMeasures= #{@previousMeasures}\n"
    description += "@measureCounter= #{@measureCounter}\n"
    description += "@firstEpoch= #{@firstEpoch}\n"
    return description
  end
end

