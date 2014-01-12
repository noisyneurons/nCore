require_relative 'Utilities'

def slope
  1.0 / 5.0
end


def ioFunction(x)
  case
    when x >= 5.0
      0.5
    when x >= 2.5
      (-1.0 * slope * x)  + 1.5
    when x >= - 2.5
      (slope * x) + 0.5
    when x >= -5.0
      (-1.0 * slope * x)  - 0.5
    else
     0.5
  end
end



puts ioFunction(-6.0)
puts ioFunction(-5.0)
puts ioFunction(-4.0)
puts ioFunction(-3.0)

puts ioFunction(-2.5)

puts ioFunction(-1.0)

puts ioFunction(0.0)

puts ioFunction(1.0)

puts ioFunction(2.5)

puts ioFunction(3.0)

puts ioFunction(4.0)
puts ioFunction(5.0)
puts ioFunction(6.0)



def ioDerivativeFromNetInput(x)
  case
    when x >= 5.0
      0.0
    when x >= 2.5
      -1.0 * slope
    when x >= - 2.5
      slope
    when x >= -5.0
      -1.0 * slope
    else
      0.0
  end
end


puts "\n\n"

puts ioDerivativeFromNetInput(-6.0)
puts ioDerivativeFromNetInput(-5.0)
puts ioDerivativeFromNetInput(-4.0)
puts ioDerivativeFromNetInput(-3.0)

puts ioDerivativeFromNetInput(-2.5)

puts ioDerivativeFromNetInput(-1.0)

puts ioDerivativeFromNetInput(0.0)

puts ioDerivativeFromNetInput(1.0)

puts ioDerivativeFromNetInput(2.5)

puts ioDerivativeFromNetInput(3.0)

puts ioDerivativeFromNetInput(4.0)
puts ioDerivativeFromNetInput(5.0)
puts ioDerivativeFromNetInput(6.0)

