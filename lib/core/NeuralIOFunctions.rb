module IOFunctionUtils
  def findNetInputThatGeneratesMaximumOutput
    gridDivisions = 20.0
    offset = 0.0
    increment = 5.0 / gridDivisions
    maxXY = nil
    40.times do
      testXs = genTestingArray(offset, increment, gridDivisions)
      maxXY = testXs.max { |a, b| a[1] <=> b[1] }
      offset = maxXY[0] - increment
      increment = 2.0 * increment / gridDivisions
    end
    maxXY[0]
  end

  private
  def genTestingArray(offset, increment, gridDivisions)
    testXs = (0..gridDivisions.to_i).collect do |aValue|
      x = offset + (aValue * increment)
      y = ioFunction(x)
      [x, y]
    end
  end
end


module SigmoidIOFunction
  include IOFunctionUtils

  def ioFunction(aNetInput)
    1.0/(1.0 + Math.exp(-1.0 * aNetInput))
  end

  def ioDerivativeFromNetInput(aNetInput)
    ioDerivativeFromOutput(ioFunction(aNetInput))
  end

  def ioDerivativeFromOutput(neuronsOutput)
    (neuronsOutput * (1.0 - neuronsOutput))
  end
end

module NonMonotonicIOFunction
  include IOFunctionUtils

  def ioFunction(x)
    1.49786971589547 * (i(x) - 0.166192596930178)
  end

  def i(x)
    h(x, 4)
  end

  def h(x, s)
    0.5 + f(x) + (-0.5 * (f(x + s) + f(x-s)))
  end

  def f(x)
    1.0 / (1.0 + Math.exp(-1.0 * x))
  end

  def ioDerivativeFromNetInput(aNetInput)
    1.49786971589547 * j(aNetInput, 4.0)
  end

  def j(x, s)
    g(x, 0.0) - (0.5 * (g(x, s) + g(x, (-1.0 * s))))
  end

  def g(x, s)
    Math.exp((-1.0 * x) + s) / ((Math.exp((-1.0 * x) + s)) + 1.0) ** 2.0
  end

end

module PiecewiseLinNonMonIOFunction
  include IOFunctionUtils

  def slope
    1.0 / 5.0
  end

  def ioFunction(x)
    case
      when x >= 5.0
        0.5
      when x >= 2.5
        (-1.0 * slope * x) + 1.5
      when x >= -2.5
        (slope * x) + 0.5
      when x >= -5.0
        (-1.0 * slope * x) - 0.5
      else
        0.5
    end
  end

  def ioDerivativeFromNetInput(x)
    case
      when x >= 5.0
        0.0
      when x >= 2.5
        -1.0 * slope
      when x >= -2.5
        slope
      when x >= -5.0
        -1.0 * slope
      else
        0.0
    end
  end
end

module NonMonotonicIOFunctionSymmetrical
  include IOFunctionUtils

  def ioFunction(x)
    (1.49786971589547 * (i(x) - 0.166192596930178)) - 0.5
  end

  def i(x)
    h(x, 4)
  end

  def h(x, s)
    f(x) + (-0.5 * (f(x + s) + f(x-s))) + 0.5
  end

  def f(x)
    1.0 / (1.0 + Math.exp(-1.0 * x))
  end

  def ioDerivativeFromNetInput(aNetInput)
    1.49786971589547 * j(aNetInput, 4.0)
  end

  def j(x, s)
    g(x, 0.0) - (0.5 * (g(x, s) + g(x, (-1.0 * s))))
  end

  def g(x, s)
    Math.exp((-1.0 * x) + s) / ((Math.exp((-1.0 * x) + s)) + 1.0) ** 2.0
  end

end

module LinearIOFunction
  include IOFunctionUtils

  def ioFunction(aNetInput)
    aNetInput
  end

  def ioDerivativeFromNetInput(aNetInput)
    ioDerivativeFromOutput(ioFunction(aNetInput))
  end

  def ioDerivativeFromOutput(neuronsOutput)
    1.0
  end

end

module SigmoidIOFunctionSymmetrical
  include IOFunctionUtils

  def ioFunction(aNetInput)
    2.0 * ((1.0/(1.0 + Math.exp(-1.0 * aNetInput))) - 0.5)
  end

  def ioDerivativeFromNetInput(aNetInput) # TODO speed this up.  Use sage to get the simpler analytical expression.
    ioDerivativeFromOutput(ioFunction(aNetInput))
  end

  def ioDerivativeFromOutput(neuronsOutput)
    2.0 * (neuronsOutput * (1.0 - neuronsOutput))
  end

end


module IOFunctionNotAccessibleHere
  include IOFunctionUtils

  def ioFunction(aNetInput)
    STDERR.puts "IO Function Accessible Only via LearningStrategy class"
    0.5
  end

  def ioDerivativeFromNetInput(aNetInput) # TODO speed this up.  Use sage to get the simpler analytical expression.
    STDERR.puts "IO Function Accessible Only via LearningStrategy class"
    0.0
  end

  def ioDerivativeFromOutput(neuronsOutput)
    STDERR.puts "IO Function Accessible Only via LearningStrategy class"
    0.0
  end
end
