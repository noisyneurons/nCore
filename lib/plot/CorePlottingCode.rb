### VERSION "nCore"
## ../nCore/lib/plot/CorePlottingCode.rb

require_relative '../core/Utilities'
require_relative '../../../rgplot/lib/gnuplotMOD'


# NOTES:

# POINT SIZE AND TYPE
# pointsize is to expand points
#  set pointsize 2.5
# type 'test' to see the colors and point types available
# lt is for color of the points: -1=black 1=red 2=grn 3=blue 4=purple 5=aqua 6=brn 7=orange 8=light-brn
# pt gives a particular point type: 1=diamond 2=+ 3=square 4=X 5=triangle 6=*
# postscipt: 1=+, 2=X, 3=*, 4=square, 5=filled square, 6=circle,
#            7=filled circle, 8=triangle, 9=filled triangle, etc.

############ Recent Additions to Plotting Section (uses a specialized library) #########################

def plotMSEvsEpochNumber(aLearningNetwork)
  mseVsEpochMeasurements = aLearningNetwork.measures
  std("measures=\t", mseVsEpochMeasurements)
  x = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:epochs] }
  y = mseVsEpochMeasurements.collect { |aMeasure| aMeasure[:mse] }

  #puts
  #puts "#{Dir.pwd}/../../plots/xyPlot"
  #puts
  #puts "working directory= \t#{Dir.pwd}"
  #aValue = File.expand_path File.dirname(__FILE__)
  #puts "working directory= \t#{aValue}"
  ## aPlotter = Plotter.new(title="Training Error", "Number of Epochs", "Error on Training Set", plotOutputFilenameBase = "/home/mark/Code/Ruby/NN2012/plots/xyPlot")

  aPlotter = Plotter.new(title="Training Error", "Number of Epochs", "Error on Training Set", plotOutputFilenameBase = "#{Dir.pwd}/../../plots/xyPlot")
  aPlotter.plot(x, y)
end

def plotDotsWhereOutputGtPt5(x, y, aNeuron, epochNumber)
  aPlotter = Plotter.new(title="Zero Xing for Neuron #{aNeuron.id} at epoch #{epochNumber}", "input 0", "input 1", plotOutputFilenameBase = "#{Dir.pwd}/../../plots/zeroXingPlot_N#{aNeuron.id}_E#{epochNumber}")
#  aPlotter = Plotter.new(title="Zero Xing for Neuron #{aNeuron.id} at epoch #{epochNumber}", "input 0", "input 1", plotOutputFilenameBase = "/home/mark/Code/Ruby/NN2012/plots/zeroXingPlot_N#{aNeuron.id}_E#{epochNumber}")
  aPlotter.plot(x, y)
end


############ ORIGINAL Plotting Section (uses a specialized library) #########################
class Plotter
  attr_accessor :xMax, :xMin, :yMax, :yMin, :zMax, :zMin

  include OS


  def initialize(title="an XY Plot", xLabel="x", yLabel="y", plotOutputFilenameBase = "#{Dir.pwd}/../../plots/xyPlot",
      #deviceSetup="png font arial 18 size 1024,768 xffffff x000000 x404040 xff0000 xffa500 x66cdaa xcdb5cd xadd8e6 x0000ff xdda0dd x9500d3",
      deviceSetup="png font arial 18 size 1024,768 #ffffff #000000 #404040 #ff0000 #ffa500 #66cdaa #cdb5cd #add8e6 #0000ff #dda0dd #9500d3",
      subTitle="", parameterLabel="")
    @title = title
    @xLabel = xLabel
    @yLabel = yLabel
    aFilename = (title.gsub(/\s+/, ""))[0..25] #  Simple way to generate a filename from the title...
    @plotOutputFilenameBase = plotOutputFilenameBase || ("#{Dir.pwd}/../../plots/" + aFilename)
    @deviceSetup = deviceSetup
    @subTitle = subTitle
    @parameterLabel = parameterLabel
    @plotFilename= "#{@plotOutputFilenameBase}.plt"
    @plotImageFilename = "#{@plotOutputFilenameBase}.#{@deviceSetup[0..2]}"
  end

  def Plotter.createTestPlot
    (Plotter.new).createTestPlot
  end

  ########################### 2-D plot of training and test error vs. epoch number   #################################
  def plotTrainingError(epochNumber, trainingMSE)
    @yMax, dummyVariable = determineScales(trainingMSE)
    File.open(@plotFilename, "w") do |gp|
      Gnuplot::Plot.new(gp) do |plot|
        plot.title "#{@title}"
        plot.xlabel "#{@xLabel}"
        plot.ylabel "#{@yLabel}"
        plot.key "right top"
        plot.yrange "[0.0:#{@yMax}]"
        #plot.size "square"

        plot.terminal @deviceSetup
        plot.output @plotImageFilename

        plot.data << Gnuplot::DataSet.new([epochNumber, trainingMSE]) do |ds|
          #ds.with = "points"
          ds.with = "lines lt 6"
          ds.linewidth = 3
          ds.title = "Training"
        end
      end
    end
    createImageFile()
  end

  ########################### 2-D plot of training and test error vs. epoch number   #################################
  def plotTrainingAndTestError(epochNumber, trainingMSE, testingMSE)
    @yMax, dummyVariable = determineScales(trainingMSE + testingMSE)
    File.open(@plotFilename, "w") do |gp|
      Gnuplot::Plot.new(gp) do |plot|
        plot.title "#{@title}"
        plot.xlabel "#{@xLabel}"
        plot.ylabel "#{@yLabel}"
        plot.key "right top"
        plot.yrange "[0.0:#{@yMax}]"
        #plot.size "square"

        plot.terminal @deviceSetup
        plot.output @plotImageFilename

        plot.data << Gnuplot::DataSet.new([epochNumber, trainingMSE]) do |ds|
          #ds.with = "points"
          ds.with = "lines lt 6"
          ds.linewidth = 3
          ds.title = "Training"
        end
        plot.data << Gnuplot::DataSet.new([epochNumber, testingMSE]) do |ds|
          #ds.with = "points"
          ds.with = "lines lt 1"
          ds.linewidth = 3
          ds.title = "Testing"
        end
      end
    end
    createImageFile()
  end

  ########################### Similar 2-D plots: Variables vs. epoch number   #################################
  def plotHyperplaneMeasuresVsEpoch(epochNumber, angleInRadians, averageOfAbsoluteWeights)
    @yMax, @yMin = determineScales(angleInRadians + averageOfAbsoluteWeights)
    File.open(@plotFilename, "w") do |gp|
      Gnuplot::Plot.new(gp) do |plot|
        plot.title "#{@title}"
        plot.xlabel "#{@xLabel}"
        plot.ylabel "#{@yLabel}"
        plot.key "right top"
        plot.yrange "[#{@yMin}:#{@yMax}]"
        #plot.size "square"

        plot.terminal @deviceSetup
        plot.output @plotImageFilename

        plot.data << Gnuplot::DataSet.new([epochNumber, angleInRadians]) do |ds|
          #ds.with = "points"
          ds.with = "lines lt 6"
          ds.linewidth = 3
          ds.title = "Angle"
        end
        plot.data << Gnuplot::DataSet.new([epochNumber, averageOfAbsoluteWeights]) do |ds|
          #ds.with = "points"
          ds.with = "lines lt 1"
          ds.linewidth = 3
          ds.title = "Weight"
        end
      end
    end
    createImageFile()
  end

  #############################################################################################################
  def plotWeightMagnitudeVsEpoch(epochNumber, beginningWeightMagnitude, endingWeightMagnitude)
    @yMax, @yMin = determineScales(beginningWeightMagnitude + endingWeightMagnitude)
    File.open(@plotFilename, "w") do |gp|
      Gnuplot::Plot.new(gp) do |plot|
        plot.title "#{@title}"
        plot.xlabel "#{@xLabel}"
        plot.ylabel "#{@yLabel}"
        plot.key "right top"
        plot.yrange "[#{@yMin}:#{@yMax}]"

        plot.terminal @deviceSetup
        plot.output @plotImageFilename

        plot.data << Gnuplot::DataSet.new([epochNumber, beginningWeightMagnitude]) do |ds|
          #ds.with = "points"
          ds.with = "lines lt 6"
          ds.linewidth = 3
          ds.title = "Beginning"
        end
        plot.data << Gnuplot::DataSet.new([epochNumber, endingWeightMagnitude]) do |ds|
          #ds.with = "points"
          ds.with = "lines lt 1"
          ds.linewidth = 3
          ds.title = "Ending"
        end
      end
    end
    createImageFile()
  end

  #############################################################################################################
  def plotNetInputVsEpoch(epochNumber, initialNetInput, netInputAfterBP, netInputAfterBPAndFlocking)
    @yMax, @yMin = determineScales(initialNetInput + netInputAfterBP + netInputAfterBPAndFlocking)
    File.open(@plotFilename, "w") do |gp|
      Gnuplot::Plot.new(gp) do |plot|
        plot.title "#{@title}"
        plot.xlabel "#{@xLabel}"
        plot.ylabel "#{@yLabel}"
        plot.key "right top"
        plot.yrange "[#{@yMin}:#{@yMax}]"

        plot.terminal @deviceSetup
        plot.output @plotImageFilename

        plot.data << Gnuplot::DataSet.new([epochNumber, initialNetInput]) do |ds|
          ds.with = "lines lt 0"
          ds.linewidth = 6
          ds.title = "Start"
        end
        plot.data << Gnuplot::DataSet.new([epochNumber, netInputAfterBP]) do |ds|
          ds.with = "lines lt 1"
          ds.linewidth = 3
          ds.title = "After BP"
        end
        plot.data << Gnuplot::DataSet.new([epochNumber, netInputAfterBPAndFlocking]) do |ds|
          ds.with = "lines lt 9"
          ds.linewidth = 3
          ds.title = "After BP+Flocking"
        end
      end
    end
    createImageFile()
  end

  #############################################################################################################
  def plotChangeInNetInputVsEpoch(epochNumber, changeFromInitialToAfterBP, changeFromInitialToAfterBPAndFlocking, logY=false)
    @yMax, @yMin = determineScales(changeFromInitialToAfterBP + changeFromInitialToAfterBPAndFlocking, addSpacer = !logY)
    File.open(@plotFilename, "w") do |gp|
      Gnuplot::Plot.new(gp) do |plot|
        plot.title "#{@title}"
        plot.xlabel "#{@xLabel}"
        plot.ylabel "#{@yLabel}"
        plot.key "right top"
        plot.yrange "[#{@yMin}:#{@yMax}]"
        plot.logscale "y" if (logY)

        plot.terminal @deviceSetup
        plot.output @plotImageFilename

        plot.data << Gnuplot::DataSet.new([epochNumber, changeFromInitialToAfterBP]) do |ds|
          ds.with = "lines lt 1"
          ds.linewidth = 3
          ds.title = "After BP"
        end
        plot.data << Gnuplot::DataSet.new([epochNumber, changeFromInitialToAfterBPAndFlocking]) do |ds|
          ds.with = "lines lt 6"
          ds.linewidth = 3
          ds.title = "After BP+Flocking"
        end
      end
    end
    createImageFile()
  end

  #############################################################################################################
  def plotDispersionVsEpoch(epochNumber, beginningDispersion, endingDispersion)
    @yMax, @yMin = determineScales(beginningDispersion + endingDispersion)
    minForLogPlot = (beginningDispersion + endingDispersion).min
    File.open(@plotFilename, "w") do |gp|
      Gnuplot::Plot.new(gp) do |plot|
        plot.title "#{@title}"
        plot.xlabel "#{@xLabel}"
        plot.ylabel "#{@yLabel}"
        plot.key "right top"
        #plot.yrange "[#{@yMin}:#{@yMax}]"
        plot.yrange "[#{minForLogPlot}:#{@yMax}]"
        plot.logscale "y"

        plot.terminal @deviceSetup
        plot.output @plotImageFilename

        plot.data << Gnuplot::DataSet.new([epochNumber, beginningDispersion]) do |ds|
          #ds.with = "points"
          ds.with = "lines lt 6"
          ds.linewidth = 3
          ds.title = "Beginning"
        end
        plot.data << Gnuplot::DataSet.new([epochNumber, endingDispersion]) do |ds|
          #ds.with = "points"
          ds.with = "lines lt 1"
          ds.linewidth = 3
          ds.title = "Ending"
        end
      end
    end
    createImageFile()
  end

  ########################### 2-D and 2-D Vector Plotting, with Animation if desired #################################
  def plot(x, y)
    createAnimatedGnuplotFile([x], [y])
    createImageFile()
  end

  def plotWithVectors(x, y, deltax, deltay)
    createAnimatedGnuplotFile([x], [y], [deltax], [deltay])
    createImageFile()
  end

  def plotAnAnimation(x, y, deltax=nil, deltay=nil) ## assuming the argument x is really an array of x arrays.  ditto for y
    createAnimatedGnuplotFile(x, y, deltax, deltay)
    createImageFile()
  end

  def plotAnAnimationWithFixedScales(x, y, deltax=nil, deltay=nil)
    x, y, deltax, deltay = scaleForPlotting(x, y, deltax, deltay)
    plotAnAnimation(x, y, deltax, deltay)
  end

  def scaleForPlotting(x, y, deltax=nil, deltay=nil)
    if (deltax.nil?)
      @xMax = x.flatten.max
      @xMin = x.flatten.min
      @yMax = y.flatten.max
      @yMin = y.flatten.min
    else
      flattenedX = x.flatten
      xVec = Vector.elements(flattenedX)
      deltaxVec = Vector.elements(deltax.flatten)
      flattenedNewX = (xVec + deltaxVec).to_a
      bothOldAndNewX = flattenedX + flattenedNewX
      @xMax = bothOldAndNewX.max
      @xMin = bothOldAndNewX.min

      flattenedY = y.flatten
      yVec = Vector.elements(flattenedY)
      deltayVec = Vector.elements(deltay.flatten)
      flattenedNewY = (yVec + deltayVec).to_a
      bothOldAndNewY = flattenedY + flattenedNewY
      @yMax = bothOldAndNewY.max
      @yMin = bothOldAndNewY.min
    end
    xRange = @xMax - @xMin
    yRange = @yMax - @yMin
    xSpacer = xRange * 0.05
    ySpacer = yRange * 0.05
    @xMax += xSpacer
    @xMin += -1.0 * xSpacer
    @yMax += ySpacer
    @yMin += -1.0 * ySpacer
    return [x, y, deltax, deltay]
  end

  def plotAColorCodedAnimationWithFixedScales(xMeasurementsForEachEpoch, yMeasurementsForEachEpoch, clusterNumberForEachEpoch)
    scaleForPlotting(xMeasurementsForEachEpoch, yMeasurementsForEachEpoch)
    createColorCodedAnimatedGnuplotFile(xMeasurementsForEachEpoch, yMeasurementsForEachEpoch, clusterNumberForEachEpoch)
    createImageFile()
  end

  #def createImageFileWindows
  #  currentDirectory = "/home/mark/Code/Ruby/NN2012"
  #  Dir.chdir(currentDirectory) do
  #    ENV['RB_GNUPLOT'] = "#{currentDirectory}/gnuplot/bin/wgnuplot_pipes.exe"
  #    plotProgramToRun = "#{currentDirectory}/gnuplot/bin/wgnuplot_pipes.exe"
  #    system("#{plotProgramToRun} #{@plotFilename}")
  #  end
  #  return @plotImageFilename
  #end

  def createImageFile
    baseDirectory = "#{Dir.pwd}/../../"
    Dir.chdir(baseDirectory) do
      plotProgramToRun = "#{baseDirectory}/gnuplot/bin/wgnuplot_pipes.exe" if (OS.windows?)
      plotProgramToRun = "gnuplot" if (OS.linux?)
      system("#{plotProgramToRun} #{@plotFilename}")
    end
    return @plotImageFilename
  end


  def createAnimatedGnuplotFile(arrayOfXArrays, arrayOfYArrays, arrayOfdeltaxArrays=nil, arrayOfdeltayArrays=nil)
    File.open(@plotFilename, "w") do |gp|
      arrayOfXArrays.each_with_index do |xArray, indexToArray|
        yArray = arrayOfYArrays[indexToArray]
        if (!arrayOfdeltaxArrays.nil?)
          arrayOfdeltaXs = arrayOfdeltaxArrays[indexToArray]
          arrayOfdeltaYs = arrayOfdeltayArrays[indexToArray]
        end

        x, y, deltax, deltay = sortPointsBasedOnXvalue(xArray, yArray, arrayOfdeltaXs, arrayOfdeltaYs)

        Gnuplot::Plot.new(gp) do |plot|
          plot.title "#{@title}"
          plot.xlabel "#{@xLabel}"
          plot.ylabel "#{@yLabel}"
          plot.key "right top"
          plot.autoscale "y" if (@yMax.nil?)
          plot.xrange "[#{@xMin}:#{@xMax}]" if (!@xMax.nil? && (indexToArray == 0))
          plot.yrange "[#{@yMin}:#{@yMax}]" if (!@yMax.nil? && (indexToArray == 0))
          plot.size "square"
          plot.xzeroaxis
          plot.yzeroaxis

          plot.terminal @deviceSetup if (indexToArray == 0)
          plot.output @plotImageFilename if (indexToArray == 0)

          if (arrayOfdeltaxArrays.nil?)
            plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
              ds.with = "points"
              #ds.with = "lines lt 8"
              #ds.linewidth = 3
              ds.title = "Label goes here"
            end
          else
            plot.data << Gnuplot::DataSet.new([x, y, deltax, deltay]) do |ds|
              ds.with = "vectors"
              ds.title = "Label goes here"
            end
          end
        end
      end
    end
    return @plotFilename
  end

  def createColorCodedAnimatedGnuplotFile(arrayOfXArrays, arrayOfYArrays, arrayOfColorCodeArrays)
    File.open(@plotFilename, "w") do |gp|
      arrayOfXArrays.each_with_index do |xArray, indexToArray|
        yArray = arrayOfYArrays[indexToArray]
        colorCodeArray = arrayOfColorCodeArrays[indexToArray]

        x0, x1, y0, y1 = splitArraysAccordingToColorCode(xArray, yArray, colorCodeArray)

        Gnuplot::Plot.new(gp) do |plot|
          plot.title "#{@title}"
          plot.xlabel "#{@xLabel}"
          plot.ylabel "#{@yLabel}"
          plot.key "right top"
          plot.autoscale "y" if (@yMax.nil?)
          plot.xrange "[#{@xMin}:#{@xMax}]" if (!@xMax.nil? && (indexToArray == 0))
          plot.yrange "[#{@yMin}:#{@yMax}]" if (!@yMax.nil? && (indexToArray == 0))
          plot.size "square"
          plot.xzeroaxis
          plot.yzeroaxis

          plot.terminal @deviceSetup if (indexToArray == 0)
          plot.output @plotImageFilename if (indexToArray == 0)

          plot.data << Gnuplot::DataSet.new([x0, y0]) do |ds|
            ds.with = "points lt 1"
            #ds.with = "lines lt 8"
            #ds.linewidth = 3
            ds.title = "Label goes here"
          end

          plot.data << Gnuplot::DataSet.new([x1, y1]) do |ds|
            ds.with = "points lt 6"
            #ds.with = "lines lt 8"
            #ds.linewidth = 3
            ds.title = "Label goes here"
          end

        end
      end
    end
    return @plotFilename
  end

########################### Contour Plotting #################################

  def plotContour(x, y, z)
    createAnimatedContourPlot([x], [y], [z])
    createImageFile()
  end

  def plotAContourAnimation(x, y, z) # assuming the argument x is really an array of x arrays.  ditto for y
    createAnimatedContourPlot(x, y, z)
    createImageFile()
  end

  def plotAContourAnimationWithFixedScales(x, y, z)
    @xMax = x.flatten.max
    @xMin = x.flatten.min
    @yMax = y.flatten.max
    @yMin = y.flatten.min
    @zMax = z.flatten.max
    @zMin = z.flatten.min
    plotAContourAnimation(x, y, z)
  end

  def createAnimatedContourPlot(arrayOfXArrays, arrayOfYArrays, arrayOfZArrays)
    File.open(@plotFilename, "w") do |gp|
      arrayOfXArrays.each_with_index do |xArray, indexToArray|
        yArray = arrayOfYArrays[indexToArray]
        zArray = arrayOfZArrays[indexToArray]

        Gnuplot::SPlot.new(gp) do |plot|
          plot.view "map"
          plot.unset "surface"
          plot.contour "base"
          plot.cntrparam "bspline"
          plot.cntrparam "levels auto 10"
          plot.style "data lines"
          plot.title "#{@title}"
          plot.xlabel "#{@xLabel}"
          plot.ylabel "#{@yLabel}"
          plot.xrange "[#{@xMin}:#{@xMax}]" if (!@xMax.nil? && (indexToArray == 0))
          plot.yrange "[#{@yMin}:#{@yMax}]" if (!@yMax.nil? && (indexToArray == 0))
          plot.zrange "[#{@zMin}:#{@zMax}]" if (!@zMax.nil? && (indexToArray == 0))
          plot.size "square"

          plot.terminal @deviceSetup if (indexToArray == 0)
          plot.output @plotImageFilename if (indexToArray == 0)

          plot.dgrid3d "30,30 gauss 0.6,0.6"
          plot.data << Gnuplot::DataSet.new([xArray, yArray, zArray]) do |ds|
            ds.with = "lines lt 8"
            ds.linewidth = 2
            #         ds.title = "Label goes here"
          end
        end
      end
    end
    return @plotFilename
  end

####################### CLUSTER PLOTTING #####################

  def plot3AnAnimatedClusterPlotWithFixedScales(arrayOfSingleEpochClusterArrays)
    allXs = []
    allYs = []

    arrayOfSingleEpochClusterArrays.each do |clusterArray|
      clusterArray.each do |cluster|
        allXs += cluster.measures['BeforeFlocking'].collect { |p| p[0] }
        allYs += ((cluster.measures['BeforeFlocking'].collect { |p| p[1] }) + (cluster.measures['BeforeFlocking'].collect { |p| p[2] }))
      end
    end
    @xMax, @xMin = determineScales(allXs)
    @yMax, @yMin = determineScales(allYs)

    plot3ClustersAnimation(arrayOfSingleEpochClusterArrays)
  end

  def plot3ClustersAnimation(arrayOfSingleEpochClusterArrays)
    arrayOfSortedClusters = sortClustersWithinEachEpoch(arrayOfSingleEpochClusterArrays)
    createAnimated3ClusterGnuplotFile(arrayOfSortedClusters)
    createImageFile()
  end

  def createAnimated3ClusterGnuplotFile(arrayOfSingleEpochClusterArrays)
    File.open(@plotFilename, "w") do |gp|
      arrayOfSingleEpochClusterArrays.each_with_index do |singleEpochClusterArray, indexToArray|
        Gnuplot::Plot.new(gp) do |plot|
          plot.title "#{@title}"
          plot.xlabel "#{@xLabel}"
          plot.ylabel "#{@yLabel}"
          plot.xzeroaxis
          plot.key "right top"
          plot.autoscale "y" if (@yMax.nil?)
          plot.xrange "[#{@xMin}:#{@xMax}]" if (!@xMax.nil? && (indexToArray == 0))
          plot.yrange "[#{@yMin}:#{@yMax}]" if (!@yMax.nil? && (indexToArray == 0))
          plot.size "square"

          plot.terminal @deviceSetup if (indexToArray == 0)
          plot.output @plotImageFilename if (indexToArray == 0)


          pointStyleSpecArray = {'BeforeFlocking' => [5, 13], 'EndOfFlocking' => [4, 12]}
          colorSpecArray = [1, 6]

          # Plot each cluster' s points
          singleEpochClusterArray.each_with_index do |cluster, clusterNumber|
            numberAssociatedWithBPError = 0
            numberAssociatedWithFlockingError = 1
            clusterNumberToLabelInLegend = 1

            nameOfMeasure = 'BeforeFlocking'
            particularMeasures = cluster.measures[nameOfMeasure]

            x = particularMeasures.collect { |p| p[0] }
            y = particularMeasures.collect { |p| p[1] }
            # plot style specification
            colorSpec = colorSpecArray[clusterNumber]
            pointStyleSpec = pointStyleSpecArray[nameOfMeasure][numberAssociatedWithBPError]

            plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
              ds.with = "points lt #{colorSpec} pt #{pointStyleSpec} ps 1"
              ds.notitle
              ds.title = "BP Error #{nameOfMeasure}" if (clusterNumber==clusterNumberToLabelInLegend)
            end

            x = particularMeasures.collect { |p| p[0] }
            y = particularMeasures.collect { |p| p[2] }
            # plot style specification
            colorSpec = colorSpecArray[clusterNumber]
            pointStyleSpec = pointStyleSpecArray[nameOfMeasure][numberAssociatedWithFlockingError]

            plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
              ds.with = "points lt #{colorSpec} pt #{pointStyleSpec} ps 1"
              ds.notitle
              ds.title = "Flock Error #{nameOfMeasure}" if (clusterNumber==clusterNumberToLabelInLegend)
            end

            nameOfMeasure = 'EndOfFlocking'
            particularMeasures = cluster.measures[nameOfMeasure]
            x = particularMeasures.collect { |p| p[0] }
            y = particularMeasures.collect { |p| p[1] }
            # plot style specification
            colorSpec = colorSpecArray[clusterNumber]
            pointStyleSpec = pointStyleSpecArray[nameOfMeasure][numberAssociatedWithBPError]

            plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
              ds.with = "points lt #{colorSpec} pt #{pointStyleSpec} ps 3"
              ds.notitle
              ds.title = "BP Error #{nameOfMeasure}" if (clusterNumber==clusterNumberToLabelInLegend)
            end

            x = particularMeasures.collect { |p| p[0] }
            y = particularMeasures.collect { |p| p[2] }
            # plot style specification
            colorSpec = colorSpecArray[clusterNumber]
            pointStyleSpec = pointStyleSpecArray[nameOfMeasure][numberAssociatedWithFlockingError]

            plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
              ds.with = "points lt #{colorSpec} pt #{pointStyleSpec} ps 3"
              ds.notitle
              ds.title = "Flock Error #{nameOfMeasure}" if (clusterNumber==clusterNumberToLabelInLegend)
            end

            # plotting vertical line to represent center of cluster's netInput
            nameOfMeasure = 'CenterOfFlock'
            xCenterOfFlock = cluster.measures[nameOfMeasure][0][0]

            x = [xCenterOfFlock, xCenterOfFlock] # particularMeasures.collect { |p| p[0] }
            y = [@yMin, @yMax] # particularMeasures.collect { |p| p[2] }
            # plot style specification
            colorSpec = colorSpecArray[clusterNumber]
            #pointStyleSpec = pointStyleSpecArray[nameOfMeasure][numberAssociatedWithFlockingError]

            plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
              ds.with = "lines lt #{colorSpec} "
              ds.notitle
              ds.notitle
            end

          end
        end
      end
    end
    return @plotFilename
  end

####################### Utility Routines #####################

  def createTestPlot
    regularWorkingDirectory = Dir.getwd
    Dir.chdir("~/Code/Ruby/NN2012") do
      testPlotFilename= "#{Dir.pwd}/../../plots/TestPlot.plt"
      currentDirectory = Dir.pwd
      puts "#{currentDirectory}"
      ENV['RB_GNUPLOT'] = "#{currentDirectory}/gnuplot/bin/wgnuplot_pipes.exe"
      plotProgramToRun = "#{currentDirectory}/gnuplot/bin/wgnuplot_pipes.exe"
      system("#{plotProgramToRun} #{testPlotFilename}")
    end
    return @plotImageFilename
  end

  def print
    puts "Printing xy Array to Plot"
    @xyValues.each { |pair| puts "#{pair[0]}, #{pair[1]}" }
  end

  def sortClustersWithinEachEpoch(arrayOfSingleEpochClusterArrays)
    arrayOfArrays = []
    arrayOfSingleEpochClusterArrays.each do |arrayOfClusters|
      sortedClusters = arrayOfClusters.sort
      arrayOfArrays << sortedClusters
    end
    return arrayOfArrays
  end

  def determineScales(allValues, addSpacer = true)
    valueMax = allValues.max
    valueMin = allValues.min
    return [valueMax, valueMin] if (addSpacer == false)
    range = valueMax - valueMin
    spacer = range * 0.05
    valueMax += spacer
    valueMin += -1.0 * spacer
    return [valueMax, valueMin]
  end

# The following routine determines whether the data involves 2 or 4 arrays and calls routines accordingly
  def sortPointsBasedOnXvalue(xArray, yArray, deltaxArray=nil, deltayArray=nil)
    if (deltaxArray.nil?)
      return sortXYPointsBasedOnXvalue(xArray, yArray)
    else
      return sortXYdeltaXdeltaYPointsBasedOnXvalue(xArray, yArray, deltaxArray, deltayArray)
    end
  end

  def sortXYPointsBasedOnXvalue(xArray, yArray)
    xyValues = xArray.zip(yArray)
    xyValues.sort! { |a, b| a[0] <=> b[0] }
    x=[]
    y=[]
    xyValues.each do |pair|
      x << pair[0]
      y << pair[1]
    end
    return [x, y]
  end

  def sortXYdeltaXdeltaYPointsBasedOnXvalue(xArray, yArray, deltaxArray, deltayArray)
    xyValues = xArray.zip(yArray).zip(deltaxArray).zip(deltayArray)
    xyValues.sort! { |a, b| a[0] <=> b[0] }
    x=[]
    y=[]
    deltax = []
    deltay = []
    xyValues.each do |pair|
      x << pair[0]
      y << pair[1]
      deltax << pair[2]
      deltay << pair[3]
    end
    return [x, y, deltax, deltay]
  end

  def splitArraysAccordingToColorCode(xArray, yArray, colorCodeArray)
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    colorCodeArray.each_with_index do |code, index|
      x0 << xArray[index] if (code==0)
      x1 << xArray[index] if (code==1)
      y0 << yArray[index] if (code==0)
      y1 << yArray[index] if (code==1)
    end
    return x0, x1, y0, y1
  end

  def reduceNumAxisLabels(xAxisLabels)
    numLabels = xAxisLabels.length
    newXAxisLabels = Array.new(numLabels, ".")
    trimFactor = (numLabels / 10) + 2
    if (trimFactor > 1)
      index = 0
      until (index > (numLabels-2))
        newXAxisLabels[index] = xAxisLabels[index]
        index += trimFactor
      end
      newXAxisLabels[-1] = xAxisLabels.last
      return newXAxisLabels
    else
      return xAxisLabels
    end
  end

end


