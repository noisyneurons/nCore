windowWidth =640
windowHeight = 640
widthForSinglePlot = 600
heightForSinglePlot = 600

Shoes.app :width => windowWidth, :height => windowHeight do
  require 'yaml'

  def originalStreamData
    @originalStreamDataRaw.collect { |anArrayOfEpochs| anArrayOfEpochs.collect { |anEpochsWorthOfData| anEpochsWorthOfData.collect { |dataForAnExample| dataForAnExample } } }
  end

  def drawXAxis
    xAxisStartingX = @plotXMargin
    xAxisStartingY = @plotYMargin + @plotHeight/2
    xAxisEndingX = @plotXMargin + @plotWidth
    xAxisEndingY = xAxisStartingY
    line(xAxisStartingX, xAxisStartingY, xAxisEndingX, xAxisEndingY)
  end

  def drawALine(pHeight, xMargin, yMargin, pWidth)
    xAxisStartingX = xMargin
    xAxisStartingY = yMargin + pHeight/2
    xAxisEndingX = xMargin + pWidth
    xAxisEndingY = xAxisStartingY
    line(xAxisStartingX, xAxisStartingY, xAxisEndingX, xAxisEndingY)
  end

  def drawYAxis
    yAxisStartingX = @plotXMargin + @plotWidth/2
    yAxisStartingY = @plotYMargin
    yAxisEndingX = yAxisStartingX
    yAxisEndingY = @plotYMargin + @plotHeight
    line(yAxisStartingX, yAxisStartingY, yAxisEndingX, yAxisEndingY)
  end

  def drawAxes
    drawXAxis
    drawYAxis
  end

  def drawHyperplaneLine(weight1, weight2, pHeight, xMargin, yMargin, pWidth)
    angleOfRotation = 180.0 * (Math.atan2(weight1, weight2)/ Math::PI)
    rotate angleOfRotation
    @l.remove unless (@l.nil?)
    @l = drawALine(pHeight, xMargin, yMargin, pWidth)
    rotate -1.0 * angleOfRotation
  end

  def calcScaleFactor(dimension, lengthOfAxisInPixels, streamData)
    data = []
    streamData.each { |aSetOfDataPoints| aSetOfDataPoints.each { |aDataPoint| data << aDataPoint } }
    scalars = data.collect { |xyDataItem| xyDataItem[dimension] }
    scalarsMin = scalars.min; scalarsMax = scalars.max; scalarsAbsMax = ([scalarsMin.abs, scalarsMax.abs]).max
    scaleFactor = lengthOfAxisInPixels.to_f / (2.0 * scalarsAbsMax)
  end

  def scaleIt(streamData, scaleFactor, dimension, lengthOfAxisInPixels, margin)
    halfAxisLength = lengthOfAxisInPixels.to_f / 2.0
    streamData.collect do |aSetOfDataPoints|
      aSetOfDataPoints.collect do |aDataPoint|
        intermediateValue = (scaleFactor * aDataPoint[dimension]) + halfAxisLength
        intermediateValue = lengthOfAxisInPixels - intermediateValue if (dimension == 1)
        aDataPoint[dimension] = (margin + intermediateValue).to_i
        aDataPoint
      end
    end
  end

  def scaleAccordingToSelectedData(selectedStreamData)
    scaleFactorX = calcScaleFactor(dimension=0, @plotWidth, selectedStreamData)
    streamDataA = scaleIt(originalStreamData, scaleFactorX, dimension=0, @plotWidth, @plotXMargin)
    scaleFactorY = calcScaleFactor(dimension=1, @plotHeight, selectedStreamData)
    return scaleIt(streamDataA, scaleFactorY, dimension=1, @plotHeight, @plotYMargin)
  end

  def drawOneBoxPerExample(dataItem, colorCodingArray)
    classOfTarget = dataItem[2].to_i
    if ((classOfTarget == 1) || (classOfTarget == 2))
      box = oval(dataItem[0], dataItem[1], 18)
    else
      box = rect(dataItem[0], dataItem[1], 12)
    end
    color = colorCodingArray[classOfTarget]
    box.style :fill => color, :center => true
    box
  end

  def displayOfDataForFirstFrame
    colorCodingArray = [red, yellow, green, blue, black, black, black]
    data = @streamOfDataToDisplay[0]
    drawHyperplaneLine(weight1 = data[0][3], weight2 = data[0][4], (@plotHeight/4), (@plotXMargin + (0.75 * @plotWidth)), @plotYMargin, (@plotWidth/4))
    littleBoxes = []
    data.each do |dataItem|
      littleBoxes << drawOneBoxPerExample(dataItem, colorCodingArray)
    end
    littleBoxes
  end

  def displayRemainingDataFrameByFrame(littleBoxes, numberOfEpochsBetweenStoringDBRecords)
    numberOfTimeSamples = @streamOfDataToDisplay.length
    @indexToData = 0
    @frameRate = 15.0
    @incrementInIndexToData = 1
    animate @frameRate do |frameNumber|
      @p.text = "Epoch Number = #{@indexToData * numberOfEpochsBetweenStoringDBRecords}            Neuron Number"
      data = @streamOfDataToDisplay[@indexToData]
      @p2.text = "Learning stage: #{data[0][5]}"
      @p3.text = "epochs in stage: #{data[0][6]}"
#      drawHyperplaneLine(weight1 = data[0][3], weight2 = data[0][4], @plotHeight, @plotXMargin, @plotYMargin, @plotWidth)
      drawHyperplaneLine(weight1 = data[0][3], weight2 = data[0][4], (@plotHeight/4), (@plotXMargin + (0.75 * @plotWidth)), @plotYMargin, (@plotWidth/4))
      littleBoxes.each_with_index do |box, dataIndex|
        dataPoint = data[dataIndex]
        box.move(dataPoint[0], dataPoint[1])
      end
      @indexToData += @incrementInIndexToData
      @indexToData = 0 if (@indexToData==(numberOfTimeSamples))
    end
  end

  #window :width => 640, :height => 640 do
  #  para "Okay, popped up from #{owner}"
  #end

  hashToTransfer = open("../../../data/neuronData") { |f| YAML.load(f) }
  numberOfEpochsBetweenStoringDBRecords = hashToTransfer[:numberOfEpochsBetweenStoringDBRecords]
  recordedNeuronIDs = hashToTransfer[:recordedNeuronIDs]
  allNeuronsStreamDataRaw = hashToTransfer[:arrayForShoes]

  @originalStreamDataRaw = allNeuronsStreamDataRaw[0]
  @indexToData = 0

  stack width: widthForSinglePlot do

    @plotXMargin = 20; @plotYMargin = 60
    @plotWidth = widthForSinglePlot - (2 * @plotXMargin)
    @plotHeight = heightForSinglePlot - (2 * @plotYMargin)

    drawAxes

    @streamOfDataToDisplay = scaleAccordingToSelectedData(originalStreamData)

    littleBoxes = displayOfDataForFirstFrame
    displayRemainingDataFrameByFrame(littleBoxes, numberOfEpochsBetweenStoringDBRecords)

    flow do
      button('rescale') do
        selectedStreamData = [originalStreamData[@indexToData]]
        @streamOfDataToDisplay = scaleAccordingToSelectedData(selectedStreamData)
      end

      button('OriginalScale') do
        @streamOfDataToDisplay = scaleAccordingToSelectedData(originalStreamData)
      end

      button('pause') do
        if (@incrementInIndexToData==1)
          @incrementInIndexToData = 0
        else
          @incrementInIndexToData = 1
        end
      end

      button('+1 Epoch') do
        numberOfTimeSamples = @streamOfDataToDisplay.length
        @p.text = "Epoch Number = #{@indexToData}            Neuron Number"
        data = @streamOfDataToDisplay[@indexToData]
        littleBoxes.each_with_index do |box, dataIndex|
          dataPoint = data[dataIndex]
          box.move(dataPoint[0], dataPoint[1])
        end
        @indexToData += 1
        @indexToData = 0 if (@indexToData==(numberOfTimeSamples))
      end

      button('-1 Epoch') do
        numberOfTimeSamples = @streamOfDataToDisplay.length
        @p.text = "Epoch Number = #{@indexToData}            Neuron Number"
        data = @streamOfDataToDisplay[@indexToData]
        littleBoxes.each_with_index do |box, dataIndex|
          dataPoint = data[dataIndex]
          box.move(dataPoint[0], dataPoint[1])
        end
        @indexToData -= 1
        @indexToData = 0 if (@indexToData==(numberOfTimeSamples))
        @indexToData = 0 if (@indexToData==(-1))
      end
    end

    @p = para 'Epoch Number                 Neuron Number' #, left: 20
    @p2 = para "Learning Stage:"
    @p3 = para "epochs in stage:"

    lb = list_box items: recordedNeuronIDs, choose: recordedNeuronIDs[0] do |s|
      @originalStreamDataRaw = allNeuronsStreamDataRaw[(s.text.to_i)-(recordedNeuronIDs[0])]
      @streamOfDataToDisplay = scaleAccordingToSelectedData(originalStreamData)
    end.move(300, 25)
  end
end

